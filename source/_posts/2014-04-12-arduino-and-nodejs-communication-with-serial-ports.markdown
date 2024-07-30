---
layout: post
title: "Arduino and NodeJS Communication With Serial Ports"
date: 2014-04-12 17:19
comments: true
categories:
- Arduino
- NodeJS

---

Hello everyone,
2 days ago I recieved my arduino kit that I had ordered and i'm really excited to tell you that I'll be posting more arduino tutorials from now on.

One of the cool things that i discoved that i'm sharing with you here, is communication of Arduino and NodeJS using serial ports.
Basically what we're doing is sending some data(zeros and ones) from the arduino to the land of Javascript and NodeJS.

<!-- more -->

###Circuit
This is the schematic of our cicuit, pretty simple. When you wire up things this way, by pressing the button we're able to digitally read the value of pin 2, which if the button is pressed it's 1 and if it isn't, the value is 0.

{% img /images/serialport_curcuit.png %}

###Arduino

The Arduino code is pretty straightforward and commented. Basically we say, pin number 2 is an input, then in our loop function we digitally read it's value every 100ms. If the value is HIGH(1) in our using Serial.write we write 1 into the serialport and if it's LOW(0) we write 0.

``` cpp
/*
  Writing to Serial Port
  Writes a digital input on pin 2 into the serial
  
  This example code is in the public domain.
 */

// digital pin 2 has a pushbutton attached to it.
int pushButton = 2;

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  // make the pushbutton's pin an input:
  pinMode(pushButton, INPUT);
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input pin:
  int buttonState = digitalRead(pushButton);
  // print out the state of the button into the serial port:
  if(buttonState == HIGH){
    Serial.write(1);
  }else{
    Serial.write(0);
  }
 
  // delay in between reads for stability
  delay(100);        
}

```

###NodeJS
With the help of ```serialport``` library we can easily get connected to the port that Arduino Board is connected and start our reading. From ```Tools/Serial Port``` in Arduino IDE, get the port that the board is connected. Mine is ```/dev/tty.usbmodem1421``` and yours might be different. On windows computers the port should start with ```COM```. When you had you port address, replace it mine.


In our js code, we load the libary and make a instance of it by passing our port address. Then after the port got oponed, we start listening for the data, and log it. By the way the data is a buffer so it's your choice how you're going to parse and use it. But here, just to show that these two worlds are communicating, the first index of data is 0 whenever we write 0, and 1 whenever we write 1 into the Serial in arduino(Which means anytime we press the button).


``` js

var SerialPort = require("serialport").SerialPort;
var serialport = new SerialPort("/dev/tty.usbmodem1421");
serialport.on('open', function(){
	console.log('Serial Port Opend');
	serialport.on('data', function(data){
		console.log(data[0]);
	});
});


```

After you uploaded the app to Arduino and it was running, in your teminal run ```node app.js``` and press the button and enjoy getting zeros and ones !!!!

----------------------

####Last but not least
There is an awesome javascript framework for Arduino called [Johnny-Five](https://github.com/rwaldron/johnny-five) which has really nie API and there a lot of cool projects and examples on their page which I highly recommend to check it out.


The source code of this tutorial is [here](https://github.com/DanialK/Arduino-NodeJs-Serialport) on GitHub.