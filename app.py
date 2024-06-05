from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import random
import mediapipe as mp
import time
from generate import Generate  # Pastikan modul ini tersedia
import constants  # Pastikan modul ini tersedia

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize game parameters
bird_img = cv2.imread("png-clipart-flappy-bird-pixel-art-minecraft-flappy-bird-sprite-game-text-thumbnail.png")
bird_img = cv2.resize(bird_img, (60, 60))  # Adjust the size as needed



cap = cv2.VideoCapture(0)
_, frm = cap.read()
height_ = frm.shape[0]
width_ = frm.shape[1]
gen = Generate(height_, width_)
s_init = False
s_time = time.time()
is_game_over = True

# Declarations
hand = mp.solutions.hands
hand_model = hand.Hands(max_num_hands=1)

def reset_game():
    global gen, s_init, s_time, is_game_over
    gen.points = 0
    gen.pipes = []
    s_init = False
    s_time = time.time()
    is_game_over = False
    constants.SPEED = 16
    constants.GEN_TIME = 1.2

def generate_frames():
    global s_init, s_time, is_game_over, gen, bird_img, hand_model

    while True:
        if is_game_over:
            # Wait for reset signal from client
            time.sleep(0.1)
            continue
        
        ss = time.time()
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)

        cv2.putText(frm, "score: " + str(gen.points), (width_ - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # Generate pipe every constants.GEN_TIME seconds
        if not s_init:
            s_init = True
            s_time = time.time()
        elif (time.time() - s_time) >= constants.GEN_TIME:
            s_init = False
            gen.create()

        frm.flags.writeable = False
        res = hand_model.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        frm.flags.writeable = True

        # Draw pipes and update their positions
        gen.draw_pipes(frm)
        gen.update()

        if res.multi_hand_landmarks:
            # Hand is detected
            pts = res.multi_hand_landmarks[0].landmark
            # Grabbing index finger point
            index_pt = (int(pts[8].x * width_), int(pts[8].y * height_))

            if gen.check(index_pt):
                # GAME OVER
                is_game_over = True
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
                frm = cv2.blur(frm, (10, 10))
                cv2.putText(frm, "GAME OVER! Press 'r' to replay", (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
                cv2.putText(frm, "Score: " + str(gen.points), (100, 180), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
                gen.points = 0
                socketio.emit('game_over', {'score': gen.points})
                continue

            # Bird: draw the bird image instead of a circle
            bird_height, bird_width, _ = bird_img.shape
            top_left = (index_pt[0] - bird_width // 2, index_pt[1] - bird_height // 2)
            bottom_right = (index_pt[0] + bird_width // 2, index_pt[1] + bird_height // 2)

            if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] <= width_ and bottom_right[1] <= height_:
                overlay = frm.copy()
                overlay[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = bird_img
                frm = cv2.addWeighted(overlay, 0.7, frm, 0.3, 0)

            # Commented out to not display hand landmarks
            # drawing.draw_landmarks(frm, res.multi_hand_landmarks[0], hand.HAND_CONNECTIONS)
                

            
        ret, buffer = cv2.imencode('.jpg', frm)
        frm = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frm + b'\r\n')
        
        if is_game_over:
            key_inp = cv2.waitKey(0)
            if key_inp == ord('r'):
                # is_game_over = False
                gen.pipes = []
                constants.SPEED = 16
                constants.GEN_TIME = 1.2
            else:
                cv2.destroyAllWindows()
                cap.release()
                break  


@socketio.on('reset_game')
def handle_reset_game():
    reset_game()
    emit('reset')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app)
