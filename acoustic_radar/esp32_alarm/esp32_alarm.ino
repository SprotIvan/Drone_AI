// esp32_alarm.ino — PASSIVE buzzer for the drone station (ESP8266 ESP-12E / ESP32).
//
// The Raspberry Pi decides (esp32_alarm.py, from the station's fused state);
// this board only beeps. Connect it to the Pi with a USB DATA cable.
//
// A PASSIVE buzzer has no oscillator inside: a steady voltage only clicks it
// once. It needs an alternating signal at an audible frequency. This sketch
// generates that square wave itself (micros()-timed, no tone(), no timers),
// at a frequency that can be changed at run time from the Pi.
//
// Protocol (115200 8N1, one command per line):
//   Pi -> board  "PING"        board -> Pi  "PONG ALARM v1 pin=14 hz=2700"
//   Pi -> board  "A1" / "A0"   board -> Pi  "OK A1" / "OK A0"   (beep / silence)
//   Pi -> board  "F2000"       board -> Pi  "OK F2000"          (tone, 50-10000 Hz)
//   at boot                    board -> Pi  "READY ALARM v1 pin=14 hz=2700"
// The Pi repeats A1/A0 every 0.5 s. With no valid A1/A0 for SILENCE_AFTER_MS
// the board goes silent by itself, so a crashed station cannot leave it on.
//
// WIRING (ESP-12E / NodeMCU): buzzer leg 1 -> GND, leg 2 -> 220 ohm -> D5
// (= GPIO14). An ESP8266 pin gives ~12 mA: a piezo disc is fine directly, a
// sealed "magnetic" passive buzzer (coil of ~16 ohm) needs an NPN transistor
// (D5 -> 1 kohm -> base, emitter -> GND, buzzer between 3.3V and collector).
// Arduino IDE board: "NodeMCU 1.0 (ESP-12E Module)".

#include <Arduino.h>
#if defined(ESP8266)
#include <ESP8266WiFi.h>
const int PIN_BUZZER = 14;           // D5 on NodeMCU / ESP-12E boards
#else
const int PIN_BUZZER = 25;           // ESP32
#endif

unsigned int        toneHz           = 3000;  // default, change with "F<hz>"
const unsigned long BEEP_ON_MS       = 200;   // beep pattern while alarmed
const unsigned long BEEP_OFF_MS      = 200;
const unsigned long SILENCE_AFTER_MS = 2000;  // fail-safe

bool alarmOn = false;
unsigned long lastCmdMs = 0;
unsigned long halfPeriodUs = 500000UL / 2700;
unsigned long lastEdgeUs = 0;
bool pinHigh = false;
String line;

void setFrequency(unsigned int hz) {
  toneHz = hz;
  halfPeriodUs = 500000UL / hz;
}

// One call per loop(): flips the pin every half period while `on`.
void squareWave(bool on) {
  if (!on) {
    if (pinHigh) {
      digitalWrite(PIN_BUZZER, LOW);
      pinHigh = false;
    }
    lastEdgeUs = micros();
    return;
  }
  unsigned long now = micros();
  if (now - lastEdgeUs >= halfPeriodUs) {
    lastEdgeUs += halfPeriodUs;
    if (now - lastEdgeUs >= halfPeriodUs) lastEdgeUs = now;   // fell behind
    pinHigh = !pinHigh;
    digitalWrite(PIN_BUZZER, pinHigh ? HIGH : LOW);
  }
}

void printId(const char *word) {
  Serial.print(word);
  Serial.print(" ALARM v1 pin=");
  Serial.print(PIN_BUZZER);
  Serial.print(" hz=");
  Serial.println(toneHz);
}

void handleLine(const String &s) {
  if (s == "PING") {
    printId("PONG");
  } else if (s == "A1" || s == "A0") {
    alarmOn = (s == "A1");
    lastCmdMs = millis();
    Serial.println(alarmOn ? "OK A1" : "OK A0");
  } else if (s.length() > 1 && s[0] == 'F') {
    long hz = s.substring(1).toInt();
    if (hz >= 50 && hz <= 10000) {
      setFrequency((unsigned int)hz);
      Serial.print("OK F");
      Serial.println(toneHz);
    } else {
      Serial.println("ERR F must be 50-10000");
    }
  } else {
    Serial.println("ERR unknown command");
  }
}

void setup() {
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_BUZZER, LOW);
#if defined(ESP8266)
  WiFi.mode(WIFI_OFF);               // not used; less timing jitter in the tone
  WiFi.forceSleepBegin();
#endif
  setFrequency(toneHz);
  Serial.begin(115200);
  delay(50);
  printId("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      line.trim();
      if (line.length() > 0) handleLine(line);
      line = "";
    } else if (c != '\r' && line.length() < 32) {
      line += c;
    }
  }

  unsigned long nowMs = millis();
  if (alarmOn && nowMs - lastCmdMs > SILENCE_AFTER_MS) {
    alarmOn = false;                 // the Pi went quiet: stop beeping
  }
  bool beepPhase = alarmOn && (nowMs % (BEEP_ON_MS + BEEP_OFF_MS)) < BEEP_ON_MS;
  squareWave(beepPhase);
}
