// ============================================================
//  servo_test.ino
//  Calibration sketch — use this to find the right angles
//  for your specific build before running the main sketch.
//
//  Open Serial Monitor at 9600 baud and send commands:
//    M<angle>  U<angle>  L<angle>
//  e.g. "M60" to test mouth open position
// ============================================================

#include <Servo.h>

Servo mouthServo;
Servo eyesUD;
Servo eyesLR;

void setup() {
  mouthServo.attach(9);
  eyesUD.attach(10);
  eyesLR.attach(11);

  mouthServo.write(90);
  eyesUD.write(90);
  eyesLR.write(90);

  Serial.begin(9600);
  Serial.println("=== Servo Calibration ===");
  Serial.println("Commands: M<0-180>  U<0-180>  L<0-180>");
  Serial.println("Example: M60 = open mouth");
  Serial.println("All servos at 90 (neutral)");
}

void loop() {
  if (!Serial.available()) return;

  char cmd = Serial.read();
  int angle = Serial.parseInt();
  angle = constrain(angle, 0, 180);

  if (cmd == 'M') {
    mouthServo.write(angle);
    Serial.print("Mouth -> "); Serial.println(angle);
  } else if (cmd == 'U') {
    eyesUD.write(angle);
    Serial.print("Eyes UD -> "); Serial.println(angle);
  } else if (cmd == 'L') {
    eyesLR.write(angle);
    Serial.print("Eyes LR -> "); Serial.println(angle);
  }

  // Flush newline
  while (Serial.available() && Serial.peek() < '0') Serial.read();
}
