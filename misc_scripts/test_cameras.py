import cv2
import pyudev

# Initialize the ArduCAM (adjust the following depending on your model and connection method)
# For USB Cameras, make sure to use the correct camera index
# camera_index = 0 or change to the correct index of your camera

def getSerialNumber(device):
    context = pyudev.Context()
    device_file = "/dev/video{}".format(device)
    device = pyudev.Devices.from_device_file(context, device_file)
    info = { item[0] : item[1] for item in device.items()}
    try:
        return info["ID_SERIAL_SHORT"]
    except:
        return None
    

if __name__ == "__main__":
    # add command line argument to specify camera indices
    
    import argparse
    parser = argparse.ArgumentParser(description='Test ArduCAMs')
    parser.add_argument('--cameras', type=int, nargs='+', default=[5], help='List of camera indices to test')
    args = parser.parse_args()
    
    cameras = {}

    target_fps = 60
    width = 640 #320
    height = 320  #240

    for path in args.cameras:
        # Open the camera
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print("Error: Could not open ArduCAM")
            exit()
            
        print("Serial Number: ", getSerialNumber(path))
        
        if cap.isOpened():
            # Get the native resolution of the camera
            
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_to_string = "".join([chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            print(f"Camera {path} native resolution: {original_width}x{original_height} at {original_fps} FPS with FourCC: {fourcc_to_string}")
            
            # To prevent 'select() timeout' errors with multiple cameras, use compressed format
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, target_fps)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_to_string = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            print(f"Camera {path} initialized with resolution {actual_width}x{actual_height} at {actual_fps} FPS with FourCC: {fourcc_to_string}")
            print(f"FourCC: {fourcc}")
            
            cameras[path] = cap
            print(f"Camera {path} started successfully")
        else:
            print(f"Error: Could not open camera {path}")
            

    print("Camera started successfully")


    try:
        while True:
            for index, cap in cameras.items():
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Failed to grab frame from camera {index}")
                    continue
        
                # Display the resulting frame in a separate window for each camera
                cv2.imshow(f'ArduCAM Feed {index}', frame)
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting due to interrupt")

    finally:
        # Release all cameras and close the windows
        for index, cap in cameras.items():
            cap.release()
            cv2.destroyWindow(f'ArduCAM Feed {index}')

        cv2.destroyAllWindows()
        print("Cameras released and windows closed")

