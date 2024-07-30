---
layout: post
title: "Cordova/PhoneGap Tutorial"
date: 2014-01-09 11:56
comments: true
categories: 
- PhoneGap
- Cordova
- AngularJS

---

If you are a web app developer and you care about mobile, you definitely heard of [PhoneGap](http://phonegap.com/)

> PhoneGap is a free and open source framework that allows you to create mobile apps using standardized web APIs for the platforms you care about.

####Before we ger started let me clarify that [PhoneGap](http://phonegap.com/) and [Cordova](http://cordova.apache.org/) are same thing so don't get confused !!!





I needed a small app to play with phonegap and last time I wrote the [simple note application](http://danialk.github.io/blog/2013/12/17/angularjs-note-application/) in AngularJS immediately I start converting it to a phonegap application, which wasn't that hard.


<!-- more -->

### Preparing Cordova 
First you need to install Cordova CLI, you can do this by running

```
$ npm install crodova -g
```

Now that you have Cordova installed, go to your project folder and run this 


```
$ cordova create cordova-app com.my.app MyAPP 
```

and then go into cordova-app folder

``` 
$ cd cordova-app 
```


Now all you need to do is to add the platforms that you want to make you application for !! Keep that in mind you need their SDKs instilled !!

``` 
  $ cordova platform add ios
```

### Application Source

Now we are ready to work on our source code which should be inside the ```www``` folder inside your cordova-app folder !!!

The source code is almost the same as the webapp version that I put in my last blog post, except a few changes that i'm going to explain now !!

Very first change it that we have to include Cordova library in our main html file.

``` 
<script src="cordova.js"></script> 
```

For every Cordova app we need one main script file to listen to ```deviceready``` event and initialise our application and our plugins !!
After release of IOS7 and PhoneGap 3 your application uses the whole screen which means the app starts from top of the screen that means you need to shift it down 20px if you want to have status bar while your application is running or otherwise you have to hide the statusbar!! There are a lot of ways to fix this problem but the easiest one is to install status bar plugin :

``` 
$ cordova plugin add  org.apache.cordova.statusbar 
```

and then we have to run this code in our main.js file :

``` js

function onDeviceReady() {
    StatusBar.overlaysWebView(false);
}
  
document.addEventListener('deviceready', onDeviceReady, false);

```

For more information about this plugin go [here](https://github.com/jonathannaguin/org.apache.cordova.statusbar) !!


### Build process and testing 

When everything finished all you need to do is to run build command which you can specify your desired platform !!

```
$ cordova build ios
```

You app is ready now !! Now you can open the Xcode project file inside the platforms/ios folder and play with it using the iOS simulator.

A small tip if you are developing for iOS, you can install ios-sim using homebrew and then you can run this which directly runs your application inside the simulator !!

``` 
$ brew install ios-sim 
```


``` 
$ cordova emulate ios 
```


### Conclusion 

This is simple app that helped me get my hands around phonegap, and hopefully was useful for you as well !!

You can get the source code from [HERE](https://github.com/DanialK/cordova-note-app) !!



