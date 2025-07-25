import py4j.GatewayServer;
import qupath.lib.gui.QuPathGUI;

QuPathGUI qupath = QuPathGUI.getInstance();
GatewayServer gatewayServer = new GatewayServer(qupath);
gatewayServer.start();
print('GatewayServer address: ' + gatewayServer.getAddress())