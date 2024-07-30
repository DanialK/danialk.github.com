---
layout: post
title: "Simple Authentication In NodeJS"
date: 2013-02-20 21:46
comments: true
categories:
- Node
- Express
- MongoDB

---

Hi everyone !!

Today i'm going to show you my simple authentication practice.
There is lots of good authentication modules out there which can be used as middleware like [PassportJS](http://passportjs.org/) and [Everyauth](https://github.com/bnoguchi/everyauth) and they are also very useful for integration your web site with social networks like Facebook and Twitter.
I will cover both the mentions tools in future posts but today we are going to build our authentication functionality on our own !!

Before we start I should mention that like always I used MongoDB as a database for saving and loading our users. And I integrate it with our app using Mongoose module. So make sure you have them installed or if you using my source files first run `npm install` . Also make sure you are running mongo server by running `mongod` in new shell.

# Application Parts
- Module Dependencies 
- Database and Models
- Middlewares and configurations
- Helper Functions
- Routes

<!-- more -->

### Firstly, what is the pass.js?
As you may noticed in my source file folder i'm using a pass.js as a module in my app. I could have just save username and password of each user in my database for authentication but to make our simple app look a little bet serious I used this file which I copied form TJ examples. basically what it does is for making new user if you pass user password to it using  `crypto`, Node's built-in module, it encrypt our user's password and save it to db. And if we trying to sing in the function take the entered password and encrypt it and check it with a encrypted password which is saved in database and if they are same, it would log the user into the site.
We could have don't use it but it's not a bad idea to use it !!

----------------------
####[Source Code In GitHub](https://github.com/DanialK/Simple-Authentication)


##Module Dependencies
Just like every node and express application :

``` js Module Dependencies
var express = require('express'),
    http = require('http'),
    mongoose = require('mongoose'),
    hash = require('./pass').hash;

var app = express();
```


##Database and Models
First we connect the application to our mongo server. Here I use myapp as my database.
We specify the Schema which contain username and password. Also salt and hash which are for our encrypting part.
Then class User which is our mongoose model and will write and read in "users" collection(document).

``` js Database and Models
mongoose.connect("mongodb://localhost/myapp");
var UserSchema = new mongoose.Schema({
    username: String,
    password: String,
    salt: String,
    hash: String
});

var User = mongoose.model('users', UserSchema);
``` 

##Middlewares and configurations

I passed the middle wares which we need for our app.
Every time weather we are successful in our authentication or not, our server send a message to the client. These messages are kept in `req.session.message` and the errors in `req.session.err`. We save this messages in two variables which have same name and then delete the session and make the session ready for next messages and errors. Then check if we have err or msg , we send them to client using res.locals.
Now we can access the `message` inside our template engine which is jade.

``` js Middlewares and configurations
app.configure(function () {
    app.use(express.bodyParser());
    app.use(express.cookieParser('Authentication Tutorial '));
    app.use(express.session());
    app.set('views', __dirname + '/views');
    app.set('view engine', 'jade');
});

app.use(function (req, res, next) {
    var err = req.session.error,
        msg = req.session.success;
    delete req.session.error;
    delete req.session.success;
    res.locals.message = '';
    if (err) res.locals.message = '<p class="msg error">' + err + '</p>';
    if (msg) res.locals.message = '<p class="msg success">' + msg + '</p>';
    next();
});
```
##Helper Functions
In authenticate function, basically we pass the username and password which the user will enter and a callback function. It will use hash function form pass.js and check if the enters user name exist in data base and if it is, it check the password .

requiredAuthentication is a function that we will pas as a middleware to the routes which needs authenticated user and if there isn't any authenticated user, redirect them to login page .

userExist is function that I only passed it to "/signup" route to check weather the username that the new user trying to make, already exists or not .

``` js Helper Functions

function authenticate(name, pass, fn) {
    if (!module.parent) console.log('authenticating %s:%s', name, pass);

    User.findOne({
        username: name
    },

    function (err, user) {
        if (user) {
            if (err) return fn(new Error('cannot find user'));
            hash(pass, user.salt, function (err, hash) {
                if (err) return fn(err);
                if (hash == user.hash) return fn(null, user);
                fn(new Error('invalid password'));
            });
        } else {
            return fn(new Error('cannot find user'));
        }
    });

}

function requiredAuthentication(req, res, next) {
    if (req.session.user) {
        next();
    } else {
        req.session.error = 'Access denied!';
        res.redirect('/login');
    }
}

function userExist(req, res, next) {
    User.count({
        username: req.body.username
    }, function (err, count) {
        if (count === 0) {
            next();
        } else {
            req.session.error = "User Exist"
            res.redirect("/signup");
        }
    });
}

```

##Routes
- "/" : if user authenticated, welcome her/him
- "/signup" : create a new user
- "/login" : check the user authentication
- "/profile" : is the route that only registered and authenticated user can access it


``` js Routes
app.get("/", function (req, res) {

    if (req.session.user) {
        res.send("Welcome " + req.session.user.username + "<br>" + "<a href='/logout'>logout</a>");
    } else {
        res.send("<a href='/login'> Login</a>" + "<br>" + "<a href='/signup'> Sign Up</a>");
    }
});

app.get("/signup", function (req, res) {
    if (req.session.user) {
        res.redirect("/");
    } else {
        res.render("signup");
    }
});

app.post("/signup", userExist, function (req, res) {
    var password = req.body.password;
    var username = req.body.username;

    hash(password, function (err, salt, hash) {
        if (err) throw err;
        var user = new User({
            username: username,
            salt: salt,
            hash: hash,
        }).save(function (err, newUser) {
            if (err) throw err;
            authenticate(newUser.username, password, function(err, user){
                if(user){
                    req.session.regenerate(function(){
                        req.session.user = user;
                        req.session.success = 'Authenticated as ' + user.username + ' click to <a href="/logout">logout</a>. ' + ' You may now access <a href="/restricted">/restricted</a>.';
                        res.redirect('/');
                    });
                }
            });
        });
    });
});

app.get("/login", function (req, res) {
    res.render("login");
});

app.post("/login", function (req, res) {
    authenticate(req.body.username, req.body.password, function (err, user) {
        if (user) {

            req.session.regenerate(function () {

                req.session.user = user;
                req.session.success = 'Authenticated as ' + user.username + ' click to <a href="/logout">logout</a>. ' + ' You may now access <a href="/restricted">/restricted</a>.';
                res.redirect('/');
            });
        } else {
            req.session.error = 'Authentication failed, please check your ' + ' username and password.';
            res.redirect('/login');
        }
    });
});

app.get('/logout', function (req, res) {
    req.session.destroy(function () {
        res.redirect('/');
    });
});

app.get('/profile', requiredAuthentication, function (req, res) {
    res.send('Profile page of '+ req.session.user.username +'<br>'+' click to <a href="/logout">logout</a>');
});

```

I've tried to cover important part of a authentication with node in this post. Hope you enjoy it.

In future posts I will cover authentication with Facebook or Twitter account using two popular authentication modules [PassportJS](http://passportjs.org/) and [Everyauth](https://github.com/bnoguchi/everyauth). Stay tuned !!! 

####[Source Code In GitHub](https://github.com/DanialK/Simple-Authentication)
