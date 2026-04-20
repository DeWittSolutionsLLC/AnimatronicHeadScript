// ============================================================
//  head_control.ino
//  Animatronic Head — Servo Controller
//
//  Receives single-character commands over Serial:
//    M<angle>  — mouth servo   (pin 9)
//    U<angle>  — eyes up/down  (pin 10)
//    L<angle>  — eyes left/right (pin 11)
//    R         — reset all to neutral
//
//  Example: "M60\n"  opens mouth
//           "U75\n"  eyes look up
//           "L120\n" eyes look right
//
//  TEST_MODE: uncomment the line below to echo commands without
//  moving servos. Useful when no servos are physically connected.
// ============================================================

#define TEST_MODE

#include <Servo.h>

Servo mouthServo;
Servo eyesUD;
Servo eyesLR;

// Startup neutral positions — adjust to match your build
const int MOUTH_CLOSED = 90;
const int EYES_UD_CENTER = 90;
const int EYES_LR_CENTER = 90;

void setup() {
#ifndef TEST_MODE
  mouthServo.attach(9);
  eyesUD.attach(10);
  eyesLR.attach(11);

  mouthServo.write(MOUTH_CLOSED);
  eyesUD.write(EYES_UD_CENTER);
  eyesLR.write(EYES_LR_CENTER);
#endif

  Serial.begin(9600);
  delay(100);
  while (Serial.available()) Serial.read();
  Serial.println("HEAD_READY");
}

void loop() {
  if (!Serial.available()) return;

  char cmd = Serial.read();
  int angle = Serial.parseInt();
  angle = constrain(angle, 0, 180);

  // Log raw received command
  Serial.print(">> ");
  Serial.print(cmd);
  if (cmd != 'R') Serial.print(angle);
  Serial.println();

  switch (cmd) {
    case 'M':
#ifndef TEST_MODE
      mouthServo.write(angle);
#endif
      Serial.print("M:");
      Serial.println(angle);
      break;
    case 'U':
#ifndef TEST_MODE
      eyesUD.write(angle);
#endif
      Serial.print("U:");
      Serial.println(angle);
      break;
    case 'L':
#ifndef TEST_MODE
      eyesLR.write(angle);
#endif
      Serial.print("L:");
      Serial.println(angle);
      break;
    case 'R':
#ifndef TEST_MODE
      mouthServo.write(MOUTH_CLOSED);
      eyesUD.write(EYES_UD_CENTER);
      eyesLR.write(EYES_LR_CENTER);
#endif
      Serial.println("RESET");
      break;
    default:
      break;
  }

  // Flush any remaining bytes (e.g. newline)
  while (Serial.available() && Serial.peek() < '0') Serial.read();
}
