from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client

from timbral_models import timbral_booming, timbral_brightness , timbral_depth , timbral_hardness , timbral_roughness , timbral_sharpness , timbral_warmth

VERBOSE = True
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

def send_osc(address, arg):
    message = osc_message_builder.OscMessageBuilder(address=address)
    message.add_arg(arg)
    maxMSP_client.send(message.build())

def handle_analyse(address, *args):
    # /return-analysis [booming] [brightess] [depth] [hardness] [roughness] [sharpness] [warmth]
    osc_content = ""
    file_ = ""
    if "file" in args:
        try: 
            file_ = args[args.index('file') + 1]
            osc_content = f"file {file_} "
            print(f"file: {file_}")
        except Exception as e: print(e)  

        if 'booming' in args :
            try: 
                if args[args.index('booming') + 1] == 1:
                    booming = timbral_booming(file_)
                    if VERBOSE:
                        print(f'booming: {booming}')
                    osc_content += ' booming ' + str(booming)
            except Exception as e: print(e)
        else:
            osc_content += ' booming none'

        if 'brightness' in args :
            try: 
                if args[args.index('brightness') + 1] == 1:
                    brightness = timbral_brightness(file_)
                    if VERBOSE:
                        print(f'brightness: {brightness}')
                    osc_content += ' brightness ' + str(brightness)
            except Exception as e: print(e)
        else:
            osc_content += ' brightness none'
        
        if 'depth' in args :
            try: 
                if args[args.index('depth') + 1] == 1:
                    depth = timbral_depth(file_)
                    if VERBOSE:
                        print(f'depth: {depth}')
                    osc_content += ' depth ' + str(depth)
            except Exception as e: print(e)
        else:
            osc_content += ' depth none'

        if 'hardness' in args :
            try: 
                if args[args.index('hardness') + 1] == 1:
                    hardness = timbral_hardness(file_)
                    if VERBOSE:
                        print(f'hardness: {hardness}')
                    osc_content += ' hardness ' + str(hardness)
            except Exception as e: print(e)
        else:
            osc_content += ' hardness none'

        if 'roughness' in args :
            try: 
                if args[args.index('roughness') + 1] == 1:
                    roughness = timbral_roughness(file_)
                    if VERBOSE:
                        print(f'roughness: {roughness}')
                    osc_content += ' roughness ' + str(roughness)
            except Exception as e: print(e)
        else:
            osc_content += ' roughness none'

        if 'sharpness' in args :
            try: 
                if args[args.index('sharpness') + 1] == 1:
                    sharpness = timbral_sharpness(file_)
                    if VERBOSE:
                        print(f'sharpness: {sharpness}')
                    osc_content += ' sharpness ' + str(sharpness)
            except Exception as e: print(e)
        else:
            osc_content += ' sharpness none'

        if 'warmth' in args :
            try: 
                if args[args.index('warmth') + 1] == 1:
                    warmth = timbral_warmth(file_)
                    if VERBOSE:
                        print(f'warmth: {warmth}')
                    osc_content += ' warmth ' + str(warmth)
            except Exception as e: print(e)
        else:
            osc_content += ' warmth none'
    
    send_osc("/return-analysis" , osc_content)

                     
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