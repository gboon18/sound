from pythonosc.udp_client import SimpleUDPClient

def main() -> None:
    client = SimpleUDPClient("127.0.0.1", 9000)

    # examples
    client.send_message("/tempo", 140.0)        # float
    client.send_message("/left/rev", 0.35)      # 0..1
    client.send_message("/right/cutoff", 1200.0) # Hz

if __name__ == "__main__":
    main()
