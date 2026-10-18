// Herramienta independiente: mueve el MG90S solo tras recibir MOVE <microsegundos>.
// Arduino-ESP32 3.x. Servo desacoplado de la puerta durante la calibración.
#include <Arduino.h>
#include <stdio.h>
#include <string.h>

constexpr uint8_t SERVO_PIN = 18;
constexpr uint16_t MIN_US = 1300;  // Ventana inicial conservadora y configurable.
constexpr uint16_t MAX_US = 1700;
constexpr uint16_t FIRST_US = 1500;  // Primera orden explícita del operador.
constexpr uint16_t MAX_STEP_US = 50;
constexpr uint16_t ABS_MIN_US = 1000;
constexpr uint16_t ABS_MAX_US = 2000;
constexpr uint32_t PERIOD_US = 20000;
constexpr uint32_t PWM_MAX = 65535;
static_assert(MIN_US >= ABS_MIN_US && MAX_US <= ABS_MAX_US && MIN_US < MAX_US &&
              FIRST_US >= MIN_US && FIRST_US <= MAX_US && MAX_STEP_US > 0,
              "Límites de calibración inválidos");

char lineBuffer[32];
size_t lineLength = 0;
bool servoReady = false;
uint16_t lastPulseUs = 0;

void handleLine(const char* line) {
  unsigned int pulse = 0;
  char extra = 0;
  if (strcmp(line, "STOP") == 0) {
    ledcWrite(SERVO_PIN, 0);
    lastPulseUs = 0;
    Serial.println("OK STOP");
    return;
  }
  if (sscanf(line, "MOVE %u %c", &pulse, &extra) != 1 ||
      pulse < MIN_US || pulse > MAX_US || !servoReady) {
    Serial.println("ERR: usa MOVE <us> dentro de MIN_US..MAX_US, o STOP");
    return;
  }
  if ((lastPulseUs == 0 && pulse != FIRST_US) ||
      (lastPulseUs != 0 &&
       abs(static_cast<int>(pulse) - static_cast<int>(lastPulseUs)) > MAX_STEP_US)) {
    Serial.println("ERR: inicia con FIRST_US y avanza en pasos de MAX_STEP_US o menos");
    return;
  }
  const uint32_t duty = (pulse * PWM_MAX) / PERIOD_US;
  if (!ledcWrite(SERVO_PIN, duty)) {
    Serial.println("ERR PWM");
    return;
  }
  lastPulseUs = pulse;
  Serial.print("OK MOVE ");
  Serial.println(pulse);
}

void setup() {
  Serial.begin(115200);
  servoReady = ledcAttach(SERVO_PIN, 50, 16);
  if (servoReady) {
    ledcWrite(SERVO_PIN, 0);  // Sin movimiento al arrancar o reiniciar.
  }
  Serial.println("Calibración lista: MOVE <us> o STOP. No se moverá solo.");
}

void loop() {
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      lineBuffer[lineLength] = '\0';
      handleLine(lineBuffer);
      lineLength = 0;
    } else if (ch != '\r') {
      if (lineLength < sizeof(lineBuffer) - 1) {
        lineBuffer[lineLength++] = ch;
      } else {
        lineLength = 0;
      }
    }
  }
}
