---
layout: post
title: "Authentication Using PassportJS"
date: 2013-02-23 01:35
comments: true
categories: 
- NodeJS
- Express
- MongoDb
- PassportJS

---

As I promised before, today i'm going to talk about authentication with [PassportJS](http://passportjs.org/).
Passport is one of the capable node modules which can be used for local authentication or even using it for integrating with Facebook or Twitter and …

Basically in passport we set up some strategies and then pass the passport middleware into our application and then passport will take care of most of the things.

Strategies can be local for your local authentication , or Facebook for authentication with Facebook .

I'm going talk about Passport API a little bit and building a sample app which have local, Facebook and Twitter authentication. And also i'm going to use MongoDB as database of this application.
#Application Overview :

1. Installing Passport and adding to app
2. Adding passport middlewares
3. Database and Models
4. Setting the strategies 
5. Function Handlers 
6. Routes 

---------

<!-- more -->
### Firstly, what is the pass.js?
As you may noticed in my source file folder i'm using a pass.js as a module in my app. I could have just save username and password of each user in my database for authentication but to make our simple app look a little bit serious I used this file which I copied form TJ examples. basically what it does is for making new user if you pass user password to it using  `crypto`, Node's built-in module, it encrypt our user's password and save it to db. And if we trying to sing in the function take the entered password and encrypt it and check it with a encrypted password which is saved in database and if they are same, it would log the user into the site.
We could have don't use it but it's not a bad idea to use it !!



-----
#Lets Get Started

###Installing Passport: 
Firstly you should install the PassportJS using NPM and require it in to your application.



	$ npm install passport
 

Because we are going to define Facebook and Twitter strategies, lets install their plugins as well as local plugin for local authentication :
 
 
	$ npm install passport-local
	$ npm install passport-facebook
 

And require them :

``` js
passport = require("passport");
LocalStrategy = require('passport-local').Strategy;
FacebookStrategy = require('passport-facebook').Strategy;

```
###Middlewares: 
Then make sure you have this middleware in use :

``` js

app.use(express.cookieParser());
app.use(express.bodyParser());
app.use(express.session({ secret: 'SECRET' }));
app.use(passport.initialize());
app.use(passport.session());

```
    

###Database and Models:

I'm using mongoose to use my MongoDB server in application.

1. Run mongo server by ``` mongod```
2. Connect to server in application``` mongoose.connect("mongodb://localhost/myapp"); ```
3. Our local user schema and it's model to store information of our local users.
``` js
	
var LocalUserSchema = new mongoose.Schema({
    username: String,
    salt: String,
    hash: String
});

var Users = mongoose.model('userauths', localUserSchema);
	
```
4. Our Facebook users schema and model to store information of the users that logged in with Facebook

``` js
var FacebookUserSchema = new mongoose.Schema({
    fbId: String,
    email: { type : String , lowercase : true},
    name : String
});
var FbUsers = mongoose.model('fbs',FacebookUserSchema);

```

###Strategies:
To use passport as our middleware first we should set the strategies to be initialised using passport initialise middleware.


####Local Strategy


``` js 

passport.use(new LocalStrategy(function(username, password,done){
    Users.findOne({ username : username},function(err,user){
        if(err) { return done(err); }
        if(!user){
            return done(null, false, { message: 'Incorrect username.' });
        }

        hash( password, user.salt, function (err, hash) {
            if (err) { return done(err); }
            if (hash == user.hash) return done(null, user);
            done(null, false, { message: 'Incorrect password.' });
        });
    });
}));

```

####Facebook Strategy
Firstly you should go to [Facebook Developer](https://developers.facebook.com/) and make a new application to get clientID and clientSecret. It should be noted that because we are running this app locally so set the app URL to ```http://localhost:3000``` . Also in callbackURL we passed the "/auth/facebook/callback" route at the end of the address because the authentication data would send to this route to be connected to Facebook and be integrated.


``` js
passport.use(new FacebookStrategy({
    clientID: "YOUR ID",
    clientSecret: "YOUR CODE",
    callbackURL: "http://localhost:3000/auth/facebook/callback"
  },
  function(accessToken, refreshToken, profile, done) {
    FbUsers.findOne({fbId : profile.id}, function(err, oldUser){
        if(oldUser){
            done(null,oldUser);
        }else{
            var newUser = new FbUsers({
                fbId : profile.id ,
                email : profile.emails[0].value,
                name : profile.displayName
            }).save(function(err,newUser){
                if(err) throw err;
                done(null, newUser);
            });
        }
    }); 
  }
));

```

The other important part of configuring passport is serializeUser and deserializeUser which basically set the user to req.user and establish a session via a cookie set in the user's browser.
``` js

passport.serializeUser(function(user, done) {
    done(null, user.id);
});


passport.deserializeUser(function(id, done) {
    FbUsers.findById(id,function(err,user){
        if(err) done(err);
        if(user){
            done(null,user);
        }else{
            Users.findById(id, function(err,user){
                if(err) done(err);
                done(null,user);
            });
        }
    });
});

```
In deserializeUser we typically find a user in database based on the given id and pass the result to done(). As you may noticed for some reason I separate the local users and Facebook users in different documents(collections) that's why I did so in deserializeUser but I believe there should be a better practice for searching a data in tow different collection that what i've done but I couldn't find such a query in mongoose api !

###Function Handlers
Like my previous example on authentication ( the simple one) here we have 2 helper function.One for check if user logged in and can access to all the parts of the site and we can use it as a middleware of the routes that need authentication,and the other one which we just use it as a middleware of post request on "/signup" route to make sure the username had not already taken .

The main different of this authenticatedOrNot function with the one in previous tutorial is that passport provide a isAuthenticated() method which we can use it intend if req.session.user which we used in previous tutorial.

``` js

function authenticatedOrNot(req, res, next){
    if(req.isAuthenticated()){
        next();
    }else{
        res.redirect("/login");
    }
}

function userExist(req, res, next) {
    Users.count({
        username: req.body.username
    }, function (err, count) {
        if (count === 0) {
            next();
        } else {
            // req.session.error = "User Exist"
            res.redirect("/singup");
        }
    });
}

```

###Routes:

The routes are quit easy to understand if you check at the source code, but there is some passport methods that i'm going to explain

First, in our post request to "/signup" request after saving a new user to database, I used passport built in login method which automatically log the new user in after signing up and redirect him/her to "/".

Second, every time you use Facebook strategy you should prepare two routes for the authentication with Facebook (or even twitter and …).

``` js

app.get("/auth/facebook", passport.authenticate("facebook",{ scope : "email"}));

app.get("/auth/facebook/callback", 
    passport.authenticate("facebook",{ failureRedirect: '/login'}),
    function(req,res){
        res.render("loggedin", {user : req.user});
    }
);

```

It use ```passport.authenticate()``` method and by passing "facebook" it knows that it should use Facebook strategy

Third, if you don't use Facebook or etc for singing in the site and you fill the login form in "/login" route and submit it, it would send a post request to "/login" which is going to be checked and authenticated by passing "local" as a first argument of ```passport.authenticate()``` method. Also there is some options like ```successRedirect``` ,``` failureRedirect ```, or even ```successFlash``` and ```failureFlash``` if you use [connect-flash](https://github.com/jaredhanson/connect-flash) .

And lastly, passport have a built in logout() method that instead of destroying session like what we did before, we can use ```req.logout()``` on our "/logout" route.

-------
#Also,
Check out the jade files and routes to see how we welcome authenticated user's and also the can access to "/profile" route.

If you have any question or even any hint to improve my knowledge I appreciate to hear from you **@DaniaL_KH** or **dani_khosravi@yahoo.com** .

Hope you enjoy this simple tutorial.

##[Source Code on Github](https://github.com/DanialK/PassportJS-Authentication)

