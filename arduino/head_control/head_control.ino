#define TEST_MODE // Comment this line out to enable physical servos

#include <Servo.h>

Servo mouthServo, eyesUD, eyesLR;

// Positions
const int MOUTH_CLOSED = 90;
int currentMouthPos = MOUTH_CLOSED;
int targetMouthPos = MOUTH_CLOSED;

// Smoothing
unsigned long lastMouthUpdate = 0;
const int mouthStepSpeed = 5; 
const int updateInterval = 15;

void setup() {
  Serial.begin(9600);
  
  #ifndef TEST_MODE
    mouthServo.attach(9);
    eyesUD.attach(10);
    eyesLR.attach(11);
    mouthServo.write(MOUTH_CLOSED);
  #else
    Serial.println("--- RUNNING IN TEST MODE (NO SERVOS) ---");
  #endif

  Serial.println("HEAD_READY");
}

void loop() {
  // 1. Listen for Commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    int angle = Serial.parseInt();
    angle = constrain(angle, 0, 180);

    if (cmd == 'M') {
      targetMouthPos = angle;
      #ifdef TEST_MODE
        Serial.print("TEST: Mouth targeting "); Serial.println(angle);
      #endif
    } else if (cmd == 'R') {
      targetMouthPos = MOUTH_CLOSED;
      #ifndef TEST_MODE
        eyesUD.write(90); eyesLR.write(90);
      #endif
    }
    
    while (Serial.available() && Serial.peek() < '0') Serial.read();
  }

  // 2. The Smoothing Logic
  if (millis() - lastMouthUpdate >= updateInterval) {
    if (currentMouthPos != targetMouthPos) {
      // Step calculation
      if (abs(currentMouthPos - targetMouthPos) <= mouthStepSpeed) {
        currentMouthPos = targetMouthPos;
      } else {
        currentMouthPos += (currentMouthPos < targetMouthPos) ? mouthStepSpeed : -mouthStepSpeed;
      }

      #ifndef TEST_MODE
        mouthServo.write(currentMouthPos);
      #endif
    }
    lastMouthUpdate = millis();
  }
}