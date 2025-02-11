from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client

local_IP = "127.0.0.1" 
receive_port = 7400  
send_port = 8800     

# Create an OSC client to send handshake request to ESP32 on its receive port
maxMSP_client = udp_client.SimpleUDPClient(local_IP, send_port)

def handle_test(address, *args):
    print(f"Received message from Max: {args}")
    testreply_message = osc_message_builder.OscMessageBuilder(address="/test-reply")
    testreply_message.add_arg("pippi")
    maxMSP_client.send(testreply_message.build())

def handle_analyse(address, *args):
    if "all" in args:
         print("received ALL")


# Function to setup and start the OSC server
def start_osc_server():
    # Create dispatcher and map the response handlers to the appropriate OSC addresses
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/test", handle_test)
    osc_dispatcher.map("/analyse", handle_analyse)
 
    # Start the OSC server, listening on port esp32_send_port
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", receive_port), osc_dispatcher)

    # Start the server to listen indefinitely
    server.serve_forever()

# Main execution entry
if __name__ == "__main__":
    start_osc_server()