---
layout: post
title: "Backbone Tips - Managing Views and Memory Leaks"
date: 2013-04-07 20:13
comments: true
categories: 
- Backbone

---

Hi everyone,

As you may see in my previous post about backbone, I used a showView for rendering my views as well as keeping track of current view and closing it.
One of the important things that you should consider in real world application is memory leaks. We always render the views and the next view comes and render it's html on previous view and we think it's gone. But it steel exist and it's listening for it's events and each new view that we render,takes our memory. So we need to somehow close our views before rendering the new one.

What we are going to do it something like this :

1. Adding close method to views prototype
2. Adding onClose method to views that have child views for closing child views
3. Adding showView to our router and using it


<!-- more -->

###Close Method :

Firstly somewhere in your code need to add close method. If you are using RequireJS for your project I think it's a good idea to require backbone in our main.js file and add it .

``` js

  Backbone.View.prototype.close = function() {
    if (this.onClose) {
      this.onClose();
    }
    this.remove();
  };


```

this code add close method to all your views and when it calls it check if view have innerViews to run onClose and get rid of them and then remove the view. remove method remove the view's element form DOM and by calling ```this.stopListening()``` unbinds all the events it's listening to. 

###onClose Method : 

Now for the views that have child views, we need to push all of them to and array and make them ready to be closed when onClose called.

``` js 

var ParentView = Backbone.View.extend({

	initialize : function () {
		this.childViews = [];
	},

	render : function(){
		this.$el.html(this.template(this.model.toJSON()));
		/*
		* adding our child views in childView array
		*/
		this.childView1 = new ChildView;
		this.childView2 = new ChildView;
		this.childViews.push(this.childView1,this.childView2)
	},

	onClose : function(){
		this.childViews.forEach(function (view){
			view.close();
		});
		/*
		
        _(this.childViews).each(function(view){
            view.close();
        });

		*/
	}
});

```

###Router's ShowView :

Now we can easily add a showView function inside our outer as a method and each time for creating an instance of our view and rendering it, use this function.
It should be noted that you can add some route filtering for checking for authentication or etc. to our showView but it is a simple one.

``` js

var Router = Backbone.Router.extend({

    routes: {
        "*": "default",
    },
    
    showView: function (view) {
        if (this.currentView) this.currentView.close();
        $('.container').html(view.$el);
        this.currentView = view;
        return view;
    },

    default: function () {
    	/*
    	* Using showView
    	*/
        this.showView(new LoginView);  
    }
});


```
So the showView, checks if there is any current view and if it is, it close it then render our view and assign the rendered view as app's currentView. At the end it returns the view that in more complex application you may need it !


Note : for view rendering and use of ```view.$el``` instead of ```view.render().el``` read my [previous backbone tip](http://danialk.github.io/blog/2013/04/07/backbone-tips-rendering-views-and-their-childviews/).


That's it.
Hope it would be helpful.


See ya !!!