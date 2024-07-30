---
layout: post
title: "Building a Contacts Manager App Using Backbonejs and Nodejs And MongoDB"
date: 2013-02-10 08:54
comments: true
categories:
- Backbone
- Node
- Express
- MongoDB
 
---

Hi guys ,
now it's a time to make our hands dirty with [Backbone](http://backbonejs.org/) and [Node.js](http://nodejs.org/) and of course [Express](http://expressjs.com/) framework !! 
Today I want to show you how to make and contacts manager with Backbone. Also I use Node.js for backend and [MongoDB](http://expressjs.com/) as a database of this project.

**[Fork this project on Github](https://github.com/DanialK/ContactsManager)**

#Quick Look 
1. Setting Up Database
2. Setting Up Server
3. Application

<!-- more -->

###Setting Up Database
As I said i'm using MongoDB of this project. Make sure you have it installed then run this code to run the database server:

{% codeblock %}
$ mongod
{% endcodeblock %}

in first parts of our server using [mongoose](http://mongoosejs.com/) we connect our server to database. It immediately make contactmanager database for you and we save all the datas in contacts collection which we set it using our mongoose model.

###Setting Up Server
Firstly make sure you have installed dependencies.you can install them by running "nom install" form the package.json that I have put in the repository.

{% codeblock app.js lang:js %}

var express = require('express')
  , mongoose =  require('mongoose')
  , http = require('http')
  , path = require('path');

var app = express();
mongoose.connect("mongodb://localhost/contactmanager");
var ContactsSchema = new mongoose.Schema({
  first_name : String,
  last_name : String,
  email_address: String,
  mobile_number : String
});
var Contacts = mongoose.model("contacts",ContactsSchema);

app.configure(function(){
  app.set('port', process.env.PORT || 3000);
  app.set('views', __dirname + '/views');
  app.set('view engine', 'jade');
  app.use(express.bodyParser());
  app.use(express.methodOverride());
  app.use(app.router);
  app.use(express.static(path.join(__dirname, 'public')));
});

app.get("/contacts", function(req,res){
  Contacts.find({},function(err,docs){
    if(err) throw err;
    res.send(docs);
  });
});

app.post("/contacts", function(req, res){
  var contact = new Contacts({
    first_name :req.body.first_name,
    last_name :req.body.last_name,
    email_address:req.body.email_address,
    mobile_number :req.body.mobile_number
  }).save(function(err,docs){
    if(err) throw err;
    res.send(docs);
  });
});

app.put("/contacts/:id", function(req,res){
  var id = req.params.id;
  Contacts.findById(id, function(err, contact) {
      if(err) throw err;
      contact.first_name = req.body.first_name,
      contact.last_name = req.body.last_name,
      contact.email_address= req.body.email_address,
      contact.mobile_number = req.body.mobile_number
      contact.save(function(err) {
        if(err) throw err;
        res.send(contact);
      });
    });
});

app.del("/contacts/:id", function(req,res){
  var id = req.params.id;
  Contacts.findById(id, function(err, contact) {
      contact.remove(function(err) {
        if(err) throw err;
        
      });
    });
});

http.createServer(app).listen(app.get('port'), function(){
  console.log("Express server listening on port " + app.get('port'));
});


{% endcodeblock %}

* As you can see in the top we connected to our database by mongoose module and also we set up all we need for storing datas for this project.
* In configuration part we add the basic configuration and call the middle wares which we need for our server. In this project I use index.html as the only view in my application and I stored it into public directory which it going to be rendered because we set the static folder to our public directory. We don't really need jade engine but I put index.jade file in views folders as well for the people who like write a new route for "/" and render it. it completely similar to our index.html file but in jade syntax.But remember you don't have to do this and you can easily ignore it and let application use index.html .
* Then we set up our routes. As you may familiar with idea of REST we have 4 routes for showing all the contact, adding contact, editing and existing contact and at last deleting a contact.



	* GET /contacts   Show All Contacts
	* POST /contacts  Add New Contact
	* PUT /contacts/:id  Edit a Contact With Id
	* Delete /contacts/:id  Delete a Contact With Id
	
	
The base url is contacts because in our application collection we set the urlRoot to "/contacts".
* At the end, like most of the times we set up the server and make it listen on the port we had set before.

###Application
In this application, I structured our models, views ,collection and router into a single js file. However as you may know in a real world application we should use some tools like JAM to make everything clean into a single js file.

####Main.js
For the sake of having a fully strutted application and doing most of the best practices, we are going to set some namespaces in our very first js file, which is main.js.

{% codeblock main.js lang:js %}

(function() {
	window.App = {
		Models: {},
		Collections: {},
		Views: {},
		Router: {}
	};

	window.vent = _.extend({}, Backbone.Events);

	window.template = function(id) {
		return _.template( $('#' + id).html() );
	};
})();

});
{% endcodeblock %}

####Router.js
There is only one route on our index which just log "INDEX" to show it's working !!

{% codeblock router.js lang:js %}

App.Router = Backbone.Router.extend({
	routes: {
		'': 'index'
	},

	index: function() {
		console.log( 'INDEX' );
	}
});

{% endcodeblock %}

####Collections.js
In collection we pass our model which we make it next and set the url for our collection :
{% codeblock collections.js lang:js %}

App.Collections.Contacts = Backbone.Collection.extend({
	model: App.Models.Contact,
	url: '/contacts'
});

{% endcodeblock %}
####Models.js
The model is contains the defaults and some validation that checks the inputs that user insert into the contact form. And also we set id attribute to "_id", because all the files that are structured into MongoDB they automatically take and "_id" attribute which contains a unique id for each contact.

{% codeblock models.js lang:js %}

App.Models.Contact = Backbone.Model.extend({
	validate: function(attrs) {
		if ( ! attrs.first_name || ! attrs.last_name ) {
			return 'A first and last name are required.';
		}

		if ( ! attrs.email_address ) {
			return 'Please enter a valid email address.';
		}

		if( isNaN(attrs.mobile_number) == true ){
			return 'Please enter a valid number';
		}
	},
	idAttribute : "_id",
	urlRoot: "/contacts",
	defaults : {
		_id: null
	}
});
{% endcodeblock %}

####Views.js
Lets take quick look at our views:

* **Global App View** - Which we set all instance of our views in it and we just run and render this view 
* **Add Contact View** - Taking all the datas from the form and insert them to collection and sync it with database(create method)
* **Edit Contact View** - Taking new datas from the edit form and put them to collation and sync 
* **All Contacts View** - loop though each contact and render it and also listing for "add" event when new contact added to update the view immediately
* **Single Contact View** - A view for a single contact which it used to render a contact and also it listen for change event to update the contact when it attributes changed and destroy event when somebody clicked the delete button and destroyed the single contact.
 
{% codeblock views.js lang:js %}

/*
|--------------------------------------------------------------------------
| Global App View
|--------------------------------------------------------------------------
*/
App.Views.App = Backbone.View.extend({
	initialize: function() {
		$("#editContact").hide();
		vent.on('contact:edit', this.editContact, this);

		var addContactView = new App.Views.AddContact({ collection: App.contacts });

		var allContactsView = new App.Views.Contacts({ collection: App.contacts });
		$('#allContacts').append(allContactsView.render().el);
	},

	editContact: function(contact) {
		var editContactView = new App.Views.EditContact({ model: contact });
		$('#editContact').html(editContactView.el);
	}
});


/*
|--------------------------------------------------------------------------
| Add Contact View
|--------------------------------------------------------------------------
*/
App.Views.AddContact = Backbone.View.extend({
	el: '#addContact',

	initialize: function() {
		this.first_name = $('#first_name');
		this.last_name = $('#last_name');
		this.mobile_number = $('#mobile_number');
		this.email_address = $('#email_address');
	},

	events: {
		'submit': 'addContact'
	},

	addContact: function(e) {
		e.preventDefault();

		this.collection.create({
			first_name: this.first_name.val(),
			last_name: this.last_name.val(),
			email_address: this.email_address.val(),
			mobile_number: this.mobile_number.val()
		}, { wait: true });
		this.clearForm();
	},

	clearForm: function() {
		this.first_name.val('');
		this.last_name.val('');
		this.mobile_number.val('');
		this.email_address.val('');
	}
});


/*
|--------------------------------------------------------------------------
| Edit Contact View
|--------------------------------------------------------------------------
*/
App.Views.EditContact = Backbone.View.extend({
	template: template('editContactTemplate'),

	initialize: function() {
		this.render();

		this.form = this.$('form');
		this.first_name = this.form.find('#edit_first_name');
		this.last_name = this.form.find('#edit_last_name');
		this.mobile_number = this.form.find('#edit_mobile_number');
		this.email_address = this.form.find('#edit_email_address');
	},

	events: {
		'submit form': 'submit',
		'click button.cancel': 'cancel'
	},

	submit: function(e) {
		e.preventDefault();

		this.model.save({
			first_name: this.first_name.val(),
			last_name: this.last_name.val(),
			mobile_number: this.mobile_number.val(),
			email_address: this.email_address.val()
		});

		this.remove();
		$("#editContact").hide();
		$("#addContact").show();
	},

	cancel: function() {
		this.remove();
		$("#editContact").hide();
		$("#addContact").show();		
	},

	render: function() {
		var html = this.template( this.model.toJSON() );

		this.$el.html(html);
		return this;
	}
});


/*
|--------------------------------------------------------------------------
| All Contacts View
|--------------------------------------------------------------------------
*/
App.Views.Contacts = Backbone.View.extend({
	tagName: 'tbody',

	initialize: function() {
		this.collection.on('add', this.addOne, this);
	},

	render: function() {
		this.collection.each( this.addOne, this );
		return this;
	},

	addOne: function(contact) {
		var contactView = new App.Views.Contact({ model: contact });
		this.$el.append(contactView.render().el);
	}
});


/*
|--------------------------------------------------------------------------
| Single Contact View
|--------------------------------------------------------------------------
*/
App.Views.Contact = Backbone.View.extend({
	tagName: 'tr',

	template: template('allContactsTemplate'),

	initialize: function() {
		this.model.on('destroy', this.unrender, this);
		this.model.on('change', this.render, this);
	},

	events: {
		'click a.delete': 'deleteContact',
		'click a.edit'  : 'editContact'
	},

	editContact: function() {
		vent.trigger('contact:edit', this.model);
		$("#addContact").hide();
		$("#editContact").show();
	},

	deleteContact: function() {
		this.model.destroy();
	},

	render: function() {
		this.$el.html( this.template( this.model.toJSON() ) );
		return this;
	},

	unrender: function() {
		this.remove();
	}
});

{% endcodeblock %}
####Lastly
at the end in script tag inside our index.html we create instance of our collection and our Global View and render them and also run our router and use backbone history to record our routes

{% codeblock lang:js %}
new App.Router;
	Backbone.history.start();

	App.contacts = new App.Collections.Contacts;
	App.contacts.fetch().then(function() {
		new App.Views.App({ collection: App.contacts });
	});
	
{% endcodeblock %}


In addition, templates play the important rule in our app. I want you take a look at two templates in our index.html file. Also I used [Twitter Bootstrap](http://twitter.github.com/bootstrap/) for styling the application.

Here link to the repository on [GitHub](https://github.com/DanialK/ContactsManager)
Hope you enjoy this article.