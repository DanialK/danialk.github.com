---
layout: post
title: "Advanced Security In Backbone Application"
date: 2013-07-28 01:45
comments: true
categories: 
- Backbone
- Express
- Authentication
- Security

---


Recently I was working on the security part of my application and I was experimenting lots of different ways for keeping the single page application secure and authorised. I saw an example of authentication in AngularJS which I found it really interesting and easy and as always(:D) started thinking how to do the same thing in my Backbone application. I tried to cover most of the advanced stuff that we need in most of web applications and usually in books and screencast about backbone there isn't mush about it and it can be nightmare for beginners !!

For the rest of this article i'm going to explain this sample application that I wrote which I tried to demonstrate route filtering, session management and securing requests using CSRF-Token in a Backbone Application.

<!-- more -->

###Source Code
Firstly I want you to download the source code and take a look at it.
[GitHub](https://github.com/DanialK/advanced-security-in-backbone)
In the rest of the article I just talk about important parts of application and highly recommend it to take a look at the source code.


#Server
I'm using express as server side framework of this sample application. Using the csrf() middleware we are adding CSRF token to all of our request and if we don't get CSRF back from the client it send an error(403) to client.
Then using jade, assign this initial CSRF token to csrf global variable in our main rendered html file that we are going to send to client.
Also when user logout, we clear the session and set new CSRF token into server's session and then send it to client.

```js



/**
 * Module dependencies.
 */

var express = require('express')
  , http = require('http')
  , path = require('path')
  , uid = require('uid2');

var app = express();

// all environments
app.set('port', process.env.PORT || 3000);
app.set('views', __dirname + '/views');
app.set('view engine', 'jade');
app.use(express.favicon());
app.use(express.logger('dev'));
app.use(express.bodyParser());
app.use(express.cookieParser('NOTHING'));
app.use(express.session());
// This middleware adds _csrf to 
// our session
// req.session._csrf
app.use(express.csrf());
app.use(express.methodOverride());
app.use(app.router);
app.use(function(req, res, next){
	res.setHeader('X-CSRF-Token', req.session._csrf);
	next();
});
app.use(express.static(path.join(__dirname, 'public')));
// development only
if ('development' == app.get('env')) {
  app.use(express.errorHandler());
}


/* ------------------------------------------------
	Application Routes
   ------------------------------------------------*/ 

app.get("/", function(req, res){
	//send and csrf token with frist request
	//and assign it to a global csrf variable
	//inside the template
	res.render('index', {
		csrf : req.session._csrf
	});
});

app.get("/session", function(req, res){ 
	//Check for authentication
	if(req.session.user){
		res.send(200, {
			auth : true,
			user : req.session.user
		});
	}else{
		res.send(401, {
			auth : false,
			csrf : req.session._csrf
		});
	}
});

app.post("/session/login", function(req, res){ 
	var email = req.body.email;
	var password = req.body.password;
	for (var i = 0; i < Users.length; i++) {
		var user = Users[i];
		if(user.email == email && user.password == password){
			req.session.user = user;
			return res.send(200, {
				auth : true,
				user : user
			});
		}
	};
	return res.send(401);
});


app.del("/session/logout", function(req, res){ 
	//Sending new csrf to client when user logged out
	//for next user to sign in without refreshing the page
	req.session.user = null;
	req.session._csrf = uid(24);

	res.send(200, {
		csrf : req.session._csrf
	});
});

app.get('/users/:id', Auth, function(req, res){
	//Using the Auth filter for this route
	//to check for authentication before sending data
	var id = req.params.id;

	for (var i = 0; i < Users.length; i++) {
		if(id == Users[i].id){
			return res.send(Users[i]);
		}
	};
	return res.send(400);
});


/* ------------------------------------------------
	Route Filters
   ------------------------------------------------*/ 

//Authentication Filter
function Auth (req, res, next) {
	if(req.session.user){
		next();
	}else{
		res.send(401,{
			flash : 'Plase log in first'
		});
	}
}


/* ------------------------------------------------
	Dummy Database
   ------------------------------------------------*/ 

var Users = [
	{
		firstName : 'Danial',
		lastName : 'Khosravi',
		password : 'pass',
		email : 'backbone@authentication.com',
		id : 1
	},
	{
		firstName : 'John',
		lastName : 'Doe',
		password : 'jd',
		email : 'john@doe.com',
		id : 2
	}
];

http.createServer(app).listen(app.get('port'), function(){
  console.log('Express server listening on port ' + app.get('port'));
});


```


#Backbone App
Application checks for authentication first it initialises and assign the result into the session model. Each api request check for the user authentication as well and sends an error if user is not authenticated. Clearing the session and sending user to login is the result of api 401 errors. This way session model is always sync with server.

##Session

In session model we keep all of our session data and login and logout functionality. Also I rewrite get, set and unset methods to use HTML5 sessionStorage if browser support it and if not, use the backbone model. The cool thing about sessionStorage is that it keep the session for you and if the session destroys, all data in it reset unlike localStorage that keep the data for you unless you delete them yourself.
csrf variable is set using the jade template when application first send to client.
Using the $.ajaxSetup we set this token to all of our future request headers. 
When user logout, server send a new CSRF token and assign it to global CSRF variable and run initialize again, so the new user don't need to refresh the page for login. 
And at the end we return an instance of model so we can require our session model in our router, views or anywhere we need it !!

```js

define([
	'jquery',
	'backbone',
	'router'
], function($, Backbone, Router){

	var SessionModel = Backbone.Model.extend({
		
		url : '/session',

		initialize : function(){
			//Ajax Request Configuration
			//To Set The CSRF Token To Request Header
			$.ajaxSetup({
				headers : {
					'X-CSRF-Token' : csrf
				}
			});

			//Check for sessionStorage support
			if(Storage && sessionStorage){
				this.supportStorage = true;
			}
		},

		get : function(key){
			if(this.supportStorage){
				var data = sessionStorage.getItem(key);
				if(data && data[0] === '{'){
					return JSON.parse(data);
				}else{
					return data;
				}
			}else{
				return Backbone.Model.prototype.get.call(this, key);
			}
		},


		set : function(key, value){
			if(this.supportStorage){
				sessionStorage.setItem(key, value);
			}else{
				Backbone.Model.prototype.set.call(this, key, value);
			}
			return this;
		},

		unset : function(key){
			if(this.supportStorage){
				sessionStorage.removeItem(key);
			}else{
				Backbone.Model.prototype.unset.call(this, key);
			}
			return this;	
		},

		clear : function(){
			if(this.supportStorage){
				sessionStorage.clear();  
			}else{
				Backbone.Model.prototype.clear(this);
			}
		},

		login : function(credentials){
			var that = this;
			var login = $.ajax({
				url : this.url + '/login',
				data : credentials,
				type : 'POST'
			});
			login.done(function(response){
				that.set('authenticated', true);
				that.set('user', JSON.stringify(response.user));
				if(that.get('redirectFrom')){
					var path = that.get('redirectFrom');
					that.unset('redirectFrom');
					Backbone.history.navigate(path, { trigger : true });
				}else{
					Backbone.history.navigate('', { trigger : true });
				}
			});
			login.fail(function(){
				Backbone.history.navigate('login', { trigger : true });
			});
		},

		logout : function(callback){
			var that = this;
			$.ajax({
				url : this.url + '/logout',
				type : 'DELETE'
			}).done(function(response){
				//Clear all session data
				that.clear();
				//Set the new csrf token to csrf vaiable and
				//call initialize to update the $.ajaxSetup 
				// with new csrf
				csrf = response.csrf;
				that.initialize();
				callback();
			});
		},


		getAuth : function(callback){
			var that = this;
			var Session = this.fetch();

			Session.done(function(response){
				that.set('authenticated', true);
				that.set('user', JSON.stringify(response.user));
			});

			Session.fail(function(response){
				response = JSON.parse(response.responseText);
				that.clear();
				csrf = response.csrf !== csrf ? response.csrf : csrf;
				that.initialize();
			});

			Session.always(callback);
		}
	});

	return new SessionModel();	
});


```


##BaseRouter
Before writing the application router, we take a look at BaseRouter. BaseRouter has before and after methods and I rewrite the route method to call before and after methods before and after changing the route!
Before has a next function as it's second argument, so when we want our application let the route handler to get executed, like node.js middlewares, we execute ```next()```.


``` js

define([
	'underscore',
	'backbone'
], function(_, Backbone){

	var BaseRouter = Backbone.Router.extend({
		before: function(){},
		after: function(){},
		route : function(route, name, callback){
			if (!_.isRegExp(route)) route = this._routeToRegExp(route);
			if (_.isFunction(name)) {
				callback = name;
				name = '';
		 	}
		  	if (!callback) callback = this[name];

		  	var router = this;

		  	Backbone.history.route(route, function(fragment) {
		   		var args = router._extractParameters(route, fragment);

		   		var next = function(){
			    	callback && callback.apply(router, args);
				    router.trigger.apply(router, ['route:' + name].concat(args));
				    router.trigger('route', name, args);
				    Backbone.history.trigger('route', router, name, args);
				    router.after.apply(router, args);		
		   		}
		   		router.before.apply(router, [args, next]);
		  	});
			return this;
		}
	});

	return BaseRouter;
});

```

##Router

Read the comments !!


``` js

define([
	'jquery',
	'underscore',
	'backbone',
	'core/BaseRouter',
	'views/HomeView',
	'views/LoginView',
	'views/ProfileView',
	'models/UserModel',
	'Session'
], function($, _,  Backbone, BaseRouter, HomeView, LoginView, ProfileView, UserModel, Session){

	var Router = BaseRouter.extend({

		routes : {
			'login' : 'showLogin',
			'profile' : 'showProfile',
			'*default' : 'showHome'
		},

		// Routes that need authentication and if user is not authenticated
		// gets redirect to login page
		requresAuth : ['#profile'],

		// Routes that should not be accessible if user is authenticated
		// for example, login, register, forgetpasword ...
		preventAccessWhenAuth : ['#login'],

		before : function(params, next){
			//Checking if user is authenticated or not
			//then check the path if the path requires authentication 
			var isAuth = Session.get('authenticated');
			var path = Backbone.history.location.hash;
			var needAuth = _.contains(this.requresAuth, path);
			var cancleAccess = _.contains(this.preventAccessWhenAuth, path);

			if(needAuth && !isAuth){
				//If user gets redirect to login because wanted to access
				// to a route that requires login, save the path in session
				// to redirect the user back to path after successful login
				Session.set('redirectFrom', path);
				Backbone.history.navigate('login', { trigger : true });
			}else if(isAuth && cancleAccess){
				//User is authenticated and tries to go to login, register ...
				// so redirect the user to home page
				Backbone.history.navigate('', { trigger : true });
			}else{
				//No problem, handle the route!!
				return next();
			}			
		},

		after : function(){
			//empty
		},

		changeView : function(view){
			//Close is a method in BaseView
			//that check for childViews and 
			//close them before closing the 
			//parentView
			function setView(view){
				if(this.currentView){
					this.currentView.close();
				}
				this.currentView = view;
				$('.container').html(view.render().$el);
			}

			setView(view);
		},

		fetchError : function(error){
			//If during fetching data from server, session expired
			// and server send 401, call getAuth to get the new CSRF
			// and reset the session settings and then redirect the user
			// to login
			if(error.status === 401){
				Session.getAuth(function(){
					Backbone.history.navigate('login', { trigger : true });
				});
			}
		},
		
		//... Route handlers …
	});

	return Router;
});

```


For a bit more specific route filtering we could use [backbone.routefilter](https://github.com/boazsender/backbone.routefilter) as well !

-------

#Conclusion
Basiclly single page web applications security management is a bit different from server side traditional websites.
In a typical application you can have these sitouations:

* Not authenticated and server the not restricted pages
* Not Authenticated and try accesing restricted page wich redirect you to login page
* Authenticated and don't have access to some pages like login, register, forgotpassowrd
* Authenticated !!!
* And leave application for a while and session expires which in a first api call to server, client get notified user is not in session anymore and redirect the user to login page


I tried to add important futures that in a real application we might need them, into this sample application. Also I highly recommend that in production serve your application on HTTPS protocol !!

You can get the source code from [GitHub](https://github.com/DanialK/advanced-security-in-backbone).

I would love to hear from you and your suggestions, feel free to leave comment.

If you enjoyed please share and fallow me on [twitter](https://twitter.com/DaniaL_KH)




