import pyrealsense2 as rs
import cv2 as cv
import numpy as np

#print(dir(rs))

point = (0, 0)
pressed = False

def mouse_callback(event, x, y, flags, param):
    global point, pressed

    if event == cv.EVENT_LBUTTONDOWN:
        pressed = True
        point = (x, y)

    elif event == cv.EVENT_MOUSEMOVE:
        if pressed:
            point = (x, y)
            
    elif event == cv.EVENT_LBUTTONUP:
        pressed = False






def realsense():




    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    


    

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())   # uint16 (mm scale)
            color_image = np.asanyarray(color_frame.get_data())   # BGR
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

            cv.imshow("Color", color_image)
            cv.imshow("Depth", depth_colormap)

            key = cv.waitKey(1)
            if key == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv.destroyAllWindows()



def realsense_distance():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    cv.namedWindow("Distance")
    cv.setMouseCallback("Distance", mouse_callback)
    try:
        while True:
            
            frames = pipeline.wait_for_frames()

          
            aligned_frames = align.process(frames)


            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

       
            if not aligned_depth_frame or not color_frame:
                continue

           
            color_image = np.asanyarray(color_frame.get_data())

            if pressed:
                x, y = point
                if 0 <= x < 640 and 0 <= y < 480:
                    dist = aligned_depth_frame.get_distance(x, y)

                    
                    cv.circle(color_image, (x, y), 5, (0, 0, 255), -1) 

                    text = f"Mesafe: {dist:.3f} m"

                    # Metni ekrana yaz (Siyah kenarlıklı beyaz yazı - okunabilirlik için)
                    cv.putText(color_image, text, (x + 10, y - 10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv.putText(color_image, text, (x + 10, y - 10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

           
            cv.imshow('Distance', color_image)

          
            key = cv.waitKey(1)
            if key & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv.destroyAllWindows()



def opencv():

    cv.namedWindow("Camera")
    cv.setMouseCallback("Camera", mouse_callback)


 

    cap = cv.VideoCapture(2, cv.CAP_DSHOW)

    if not cap.isOpened():
        exit()



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.circle(frame, point, 5, (0, 0, 255), -1)
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__=="__main__":
    realsense_distance()