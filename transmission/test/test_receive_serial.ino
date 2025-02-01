// #include "PPMEncoder.h"

//////////////////////CONFIGURATION///////////////////////////////
#define chanel_number 6          // set the number of chanels
#define default_servo_value 1500 // set the default servo value
#define PPM_FrLen 20000          // set the PPM frame length in microseconds (1ms = 1000µs)
#define PPM_PulseLen 400         // set the pulse length
#define onState 0                // set polarity of the pulses: 1 is positive, 0 is negative
#define sigPin 10                // set PPM signal output pin on the arduino
//////////////////////////////////////////////////////////////////
#define defaultSteering 0
#define defaultThrottle 0

/*this array holds the servo values for the ppm signal
 change theese values in your code (usually servo values move between 1000 and 2000)*/
int ppm[chanel_number];

void setup()
{
  Serial.begin(9600); // Initialize serial communication at 9600 baud rate

  ppm[0] = default_servo_value + (defaultSteering * 500);
  ppm[1] = default_servo_value + (defaultThrottle * 500);

  for (int i = 2; i < chanel_number; i++)
  {
    ppm[i] = default_servo_value;
  }

  pinMode(sigPin, OUTPUT);
  digitalWrite(sigPin, !onState); // set the PPM signal pin to the default state (off)

  cli();
  TCCR1A = 0; // set entire TCCR1 register to 0
  TCCR1B = 0;

  OCR1A = 100;             // compare match register, change this
  TCCR1B |= (1 << WGM12);  // turn on CTC mode
  TCCR1B |= (1 << CS11);   // 8 prescaler: 0,5 microseconds at 16mhz
  TIMSK1 |= (1 << OCIE1A); // enable timer compare interrupt
  sei();
}

void loop()
{
  if (Serial.available() > 0)
  {
    // Read the input as a string
    String input = Serial.readStringUntil('\n');

    // Parse the input string
    int spaceIndex = input.indexOf(' ');
    if (spaceIndex > 0)
    {
      // Extract the percentages from the input string

      // speed and throttle are in the range [-1, 1]
      double speed = input.substring(0, spaceIndex).toDouble();
      double throttle = input.substring(spaceIndex + 1).toDouble();

      ppm[0] = default_servo_value + (speed * 500);
      ppm[1] = default_servo_value + (throttle * 500);
    }
  }
}

ISR(TIMER1_COMPA_vect)
{ // leave this alone
  static boolean state = true;

  TCNT1 = 0;

  if (state)
  { // start pulse
    digitalWrite(sigPin, onState);
    OCR1A = PPM_PulseLen * 2;
    state = false;
  }
  else
  { // end pulse and calculate when to start the next pulse
    static byte cur_chan_numb;
    static unsigned int calc_rest;

    digitalWrite(sigPin, !onState);
    state = true;

    if (cur_chan_numb >= chanel_number)
    {
      cur_chan_numb = 0;
      calc_rest = calc_rest + PPM_PulseLen; //
      OCR1A = (PPM_FrLen - calc_rest) * 2;
      calc_rest = 0;
    }
    else
    {
      OCR1A = (ppm[cur_chan_numb] - PPM_PulseLen) * 2;
      calc_rest = calc_rest + ppm[cur_chan_numb];
      cur_chan_numb++;
    }
  }
}
