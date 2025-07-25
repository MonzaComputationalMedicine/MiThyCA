import argparse
import os
import pprint
import urllib.parse
import uuid
import warnings
from random import shuffle

import geojson
from py4j.java_gateway import JavaGateway
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

os.environ['VIPS_WARNING'] = '1'
warnings.simplefilter(action='ignore', category=FutureWarning)

from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import openslide
import skimage.color as color
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.transform as transform
import torch
import torch.nn.functional as F
from PIL import Image
from openslide import OpenSlide
from torch import nn
from torchvision.transforms.v2 import *
from tqdm import tqdm

def pil2np(img, mode='f'):
    return np.array(img) / 255 if mode == 'f' else np.array(img)


def np2pil(img, mode='f'):
    return Image.fromarray(np.uint8(img * 255 if mode == 'f' else img))


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def get_bounding_box(thumbnail, main_component=False):
    w, h = thumbnail.size
    hed_image = color.rgb2hed(thumbnail)
    hed_image[:, :, 2:] = 0
    rgb_image = color.hed2rgb(hed_image)
    gray_image = color.rgb2gray(rgb_image)
    threshold = filters.threshold_otsu(gray_image)
    mask = gray_image < threshold
    mask = morphology.binary_closing(mask, footprint=morphology.disk(3))
    if main_component:
        mask = getLargestCC(mask)
    mask = morphology.binary_dilation(mask, footprint=morphology.disk(25))
    rows, cols = np.where(mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    return np.array([min_row / h, max_row / h, min_col / w, max_col / w], dtype=np.float32), mask


def get_mask(thumbnail, main_component=False):
    hed_image = color.rgb2hed(thumbnail)
    hed_image[:, :, 2:] = 0
    rgb_image = color.hed2rgb(hed_image)
    gray_image = color.rgb2gray(rgb_image)
    threshold = filters.threshold_otsu(gray_image)
    mask = gray_image < threshold
    mask = morphology.binary_closing(mask, footprint=morphology.disk(5))
    if main_component:
        mask = getLargestCC(mask)
    # mask = morphology.binary_dilation(mask, footprint=morphology.disk(25))
    return mask


hed2rgb_mat = torch.tensor([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]], dtype=torch.float32)
rgb2hed_mat = torch.linalg.inv(hed2rgb_mat)

adjust_eps = 1e-6
log_adjust_eps = np.log(adjust_eps)


class Rgb2Hed(nn.Module):
    def forward(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        rgb_tensor = rgb_tensor.to(torch.float32)
        torch.clamp_min(rgb_tensor, adjust_eps, out=rgb_tensor)
        stains_tensor = torch.einsum('...ijk,...il->...ljk', torch.log(rgb_tensor) / log_adjust_eps, rgb2hed_mat)
        torch.clamp_min(stains_tensor, 0, out=stains_tensor)
        return stains_tensor


class Hed2Rgb(nn.Module):
    def forward(self, stains_tensor: torch.Tensor) -> torch.Tensor:
        rgb_tensor = torch.exp(torch.einsum('...ijk,...il->...ljk', stains_tensor * log_adjust_eps, hed2rgb_mat))
        torch.clip(rgb_tensor, 0, 1, out=rgb_tensor)
        return rgb_tensor


class Hed2He(nn.Module):
    def forward(self, stains_tensor: torch.Tensor) -> torch.Tensor:
        if stains_tensor.dim() == 3:
            return stains_tensor[0:2, :, :]
        return stains_tensor[:, 0:2, :, :]


class He2Hed(nn.Module):
    def forward(self, stains_tensor: torch.Tensor) -> torch.Tensor:
        sz = list(stains_tensor.size())
        sz[-3] = 1
        pad = torch.zeros(sz)
        return torch.cat((stains_tensor, pad), dim=-3)


transforms = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Rgb2Hed(),
    Hed2He(),
    He2Hed(),
    Hed2Rgb(),
])


def predict(model, batch):
    x = model(batch)
    if isinstance(x, ImageClassifierOutputWithNoAttention):
        x = x.logits
    return F.softmax(x, -1)


def get_config_for_mpp(slide, mpp, size):
    base_mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    base_mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    base_mpp = (base_mpp_x + base_mpp_y) / 2
    downsample = mpp / base_mpp
    level = slide.get_best_level_for_downsample(downsample)
    extra_downsample = downsample / slide.level_downsamples[level]
    extra_size = int(size * extra_downsample)
    delta_size = int(extra_size * slide.level_downsamples[level])
    return level, (extra_size, extra_size), delta_size, lambda x: x.resize((size, size), Image.Resampling.LANCZOS)


def load_tile(slide, coord, size, level, resize):
    p, c = coord
    img = Image.new('RGB', size, (255, 255, 255))
    region = slide.read_region(c, level, size)
    img.paste(region, mask=region.split()[3])
    img = resize(img)
    img_gray = np.array(img.convert('L'))
    if np.mean(img_gray > 235) > .2:
        return None
    return p, c, transforms(img)


def compute_point_grid(shape, mask, jump, square_side_len, delta, extra_mask=None):
    coords = []
    # Iterate over the grid defined by the bounding box and the jump step size
    for i, h in enumerate(range(0, shape[0] - jump, jump)):
        for j, w in enumerate(range(0, shape[1] - jump, jump)):
            if not mask[round(h / shape[0] * mask.shape[0]), round(w / shape[1] * mask.shape[1])]:
                continue
            if not (extra_mask is None or extra_mask[i, j]):
                continue
            # Calculate the top-left corner of the centered square
            sw = int(w + jump / 2) - square_side_len // 2
            sh = int(h + jump / 2) - square_side_len // 2

            # Generate coordinates along the vertical edges of the square
            coords += [((i, j), (sw, h_))
                       for h_ in range(sh, sh + square_side_len, delta)]
            coords += [((i, j), (sw + square_side_len - delta, h_))
                       for h_ in range(sh, sh + square_side_len, delta)]

            # Generate coordinates along the horizontal edges of the square
            coords += [((i, j), (w_, sh))
                       for w_ in range(sw + delta, sw + square_side_len - delta, delta)]
            coords += [((i, j), (w_, sh + square_side_len - delta))
                       for w_ in range(sw + delta, sw + square_side_len - delta, delta)]
    shuffle(coords)
    return coords


def results_batch(jobs, batch_size, pbar=None):
    read_now = 0
    batch = []
    for future in as_completed(jobs):
        if (r := future.result()) is not None:
            batch.append(r)
        del jobs[future]
        read_now += 1
        if pbar:
            pbar.update()
        if len(batch) == batch_size:
            break
    return list(zip(*batch)), read_now


def make_geojson(slide, heatmap_1, heatmap_2, threshold_1, threshold_2, calibration=False):
    def close_scale_loop(slide, path, ratio):
        bounds = np.array(
            [slide.properties.get(f'openslide.bounds-{s}', 0) for s in ('x', 'y')],
            dtype=np.int64
        )
        path = np.int64(np.flip(path - 1/2, 1) * ratio - bounds).tolist()
        return path + [path[0]]


    heatmap_1 = heatmap_1.copy()
    heatmap_2 = heatmap_2.copy()
    if calibration:
        heatmap_1[0, 0] = heatmap_1[heatmap_1.shape[0] - 1, heatmap_1.shape[1] - 1] = 1.
        heatmap_2[0, heatmap_2.shape[1] - 1] = heatmap_2[heatmap_2.shape[0] - 1, 0] = 1.

    upscale = 20
    ratio = np.array(slide.dimensions) / np.flip(heatmap_1.shape) / upscale
    labeled_mask_1, num_features_1 = measure.label(
        # transform.rescale(heatmap_1, upscale) > threshold_1,
        filters.gaussian(transform.rescale(heatmap_1, upscale), 1) > threshold_1,
        return_num=True
    )
    labeled_mask_2, num_features_2 = measure.label(
        # transform.rescale(heatmap_2, upscale) > threshold_2,
        filters.gaussian(transform.rescale(heatmap_2, upscale), 1) > threshold_2,
        return_num=True
    )
    contours = ([(0, measure.find_contours(np.pad(labeled_mask_1 == lab + 1, 1))) for lab in range(num_features_1)] +
                [(1, measure.find_contours(np.pad(labeled_mask_2 == lab + 1, 1))) for lab in range(num_features_2)])

    features = [
        geojson.Feature(
            geometry=geojson.MultiPolygon([
                [close_scale_loop(slide, c, ratio)] for c in contour
            ]) if len(contour) > 1 else geojson.Polygon(
                [close_scale_loop(slide, contour[0], ratio)]
            ),
            properties={
                "objectType": "annotation",
                "classification": {
                    "name": "PTC-Only", "color": [255, 225, 0]
                } if i else {
                    "name": "Neoplastic", "color": [150, 255, 150]
                },
                "id": str(uuid.uuid4()),
                "isLocked": True,
            }
        ) for i, contour in contours
    ]

    return geojson.FeatureCollection(features)


def main():
    parser = argparse.ArgumentParser(description="Process configuration for WSI analysis.")
    parser.add_argument('--file', default=None, type=str, help='WSI file')
    parser.add_argument('--model_path_1', default='./model_1.pt', help='Path to the first model')
    parser.add_argument('--model_path_2', default='./model_2.pt', help='Path to the second model')
    parser.add_argument('--thumbnail_size', type=int, default=2048, help='Size of the thumbnail')
    parser.add_argument('--tile_size_1', type=int, default=96, help='Tile size for the first model')
    parser.add_argument('--tile_size_2', type=int, default=224, help='Tile size for the second model')
    parser.add_argument('--square_side_1', type=int, default=4, help='Side length of the first square')
    parser.add_argument('--square_side_2', type=int, default=2, help='Side length of the second square')
    parser.add_argument('--mpp_1', type=float, default=0.97, help='Microns per pixel for the first model')
    parser.add_argument('--mpp_2', type=float, default=0.8, help='Microns per pixel for the second model')
    parser.add_argument('--max_splits', type=int, default=200, help='Maximum number of splits')
    parser.add_argument('--block_scale', type=float, default=1.2, help='Scale of block relative to square')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--batch_size_1', type=int, default=512, help='Batch size for the first model')
    parser.add_argument('--batch_size_2', type=int, default=128, help='Batch size for the second model')
    parser.add_argument('--buffer_size_1', type=int, default=1024, help='Buffer size for the first model')
    parser.add_argument('--buffer_size_2', type=int, default=256, help='Buffer size for the second model')
    parser.add_argument('--threshold_1', type=float, default=0.8, help='First threshold value')
    parser.add_argument('--threshold_2', type=float, default=0.5, help='Second threshold value')
    parser.add_argument('--device', default='cpu', help='Device to run the computation on')

    args = parser.parse_args()

    config = vars(args)

    print("Configuration:")
    pprint.pprint(config, indent=4)

    model_1 = torch.load(args.model_path_1, weights_only=False).to(args.device).eval()
    model_2 = torch.load(args.model_path_2, weights_only=False).to(args.device).eval()

    if args.file is None:
        gateway = JavaGateway()
        path = urllib.parse.unquote(gateway.entry_point.getImageData().getServer().getURIs()[0].toString())
        assert path.startswith('file:/')
        wsi = path[5:]
    else:
        wsi = args.file
    s_name = wsi.split('/')[-1]

    slide = OpenSlide(wsi)
    level_1, size_1, delta_1, resize_1 = get_config_for_mpp(slide, args.mpp_1, args.tile_size_1)
    square_side_len_1 = delta_1 * args.square_side_1

    level_2, size_2, delta_2, resize_2 = get_config_for_mpp(slide, args.mpp_2, args.tile_size_2)
    square_side_len_2 = delta_2 * args.square_side_2

    thumbnail = slide.get_thumbnail((args.thumbnail_size, args.thumbnail_size))
    fw, fh = slide.dimensions

    mask = get_mask(thumbnail)
    jump = round(max(
        max(fw, fh) / args.max_splits,
        args.block_scale * args.square_side_1 * delta_1,
        args.block_scale * args.square_side_2 * delta_2
    ))

    heatmap_1 = np.zeros((fh // jump, fw // jump), dtype=np.float32)
    heatmap_2 = np.zeros_like(heatmap_1, dtype=np.float32)
    counts_1 = np.zeros_like(heatmap_1, dtype=np.uint64)
    counts_2 = np.zeros_like(heatmap_2, dtype=np.uint64)

    with ThreadPoolExecutor(args.max_workers) as executor:
        coords_1 = compute_point_grid((fh, fw), mask, jump, square_side_len_1, delta_1)

        jobs = {executor.submit(load_tile, slide, coord, size_1, level_1, resize_1): coord for coord in
                coords_1[:args.buffer_size_1]}
        cursor_1 = args.buffer_size_1

        pbar_in = tqdm(total=len(coords_1), desc=s_name + ' - 1')
        while len(jobs):
            batch, read_now = results_batch(jobs, args.batch_size_1, pbar=pbar_in)
            jobs.update({executor.submit(load_tile, slide, coord, size_1, level_1, resize_1): coord
                         for coord in coords_1[cursor_1:cursor_1 + read_now]})
            cursor_1 += read_now

            if len(batch):
                pts, cds, ims = batch
                ims = torch.stack(ims, dim=0).to(args.device)
                with torch.no_grad():
                    pds = predict(model_1, ims)[:, 1]
                for (x, y), c, p in zip(pts, cds, pds):
                    heatmap_1[x, y] += p
                    counts_1[x, y] += 1
        heatmap_1 /= np.maximum(1, counts_1)
        pbar_in.close()

        t_mask = heatmap_1 > args.threshold_1
        coords_2 = compute_point_grid((fh, fw), mask, jump, square_side_len_2, delta_2, extra_mask=t_mask)

        jobs = {executor.submit(load_tile, slide, coord, size_2, level_2, resize_2): coord for coord in
                coords_2[:args.buffer_size_2]}
        cursor_2 = args.buffer_size_2

        pbar_in = tqdm(total=len(coords_2), desc=s_name + ' - 2')
        while len(jobs):
            batch, read_now = results_batch(jobs, args.batch_size_2, pbar=pbar_in)
            jobs.update({executor.submit(load_tile, slide, coord, size_2, level_2, resize_2): coord
                         for coord in coords_2[cursor_2:cursor_2 + read_now]})
            cursor_2 += read_now

            if len(batch):
                pts, cds, ims = batch
                ims = torch.stack(ims, dim=0).to(args.device)
                with torch.no_grad():
                    pds = predict(model_2, ims)[:, 1]
                for (x, y), c, p in zip(pts, cds, pds):
                    heatmap_2[x, y] += p
                    counts_2[x, y] += 1
        heatmap_2 /= np.maximum(1, counts_2)
        pbar_in.close()

        del jobs

    gjson = make_geojson(slide, heatmap_1, heatmap_2, args.threshold_1, args.threshold_2)
    if args.file is None:
        annotations = gateway.jvm.com.google.gson.JsonParser.parseString(geojson.dumps(gjson))
        paths = gateway.jvm.qupath.lib.io.GsonTools.parseObjectsFromGeoJSON(annotations)
        gateway.jvm.qupath.lib.scripting.QP.clearAllObjects()
        gateway.jvm.qupath.lib.scripting.QP.addObjects(paths)
        gateway.close()
    else:
        outfile = '.'.join(s_name.split('.')[:-1]) + '.geojson'
        with open(outfile, 'w') as f:
            geojson.dump(gjson, f)


if __name__ == '__main__':
    main()
