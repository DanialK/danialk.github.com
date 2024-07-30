---
layout: post
title: "Backbone Tips: Rendering Views and Their ChildViews"
date: 2013-04-07 18:39
comments: true
categories: 
- Backbone

---

Hi everyone,
It is about more than 1 month that I haven't post anything cause I was working on a backbone project and during this time I've learned some new tricks which you need to know them if you really want to write a real application.

####First important thing is managing your childViews in parentView.

<!-- more -->

There is couple of tricks that you can use
for example render child view and then append it element to parent view's element :

``` js
...
initialize : function(){
	this.childView1 = new ChildView({});
	this.childView2 = new ChildView({});	
},
render : function(){
	this.$el.html(this.template());
	this.$el.append(this.childView1.render().el);
	this.$el.append(this.childView2.render().el);
}
...
```

or even use setElement method :

``` js
...
initialize : function(){
	this.childView1 = new ChildView({});
	this.childView2 = new ChildView({});	
},
render : function(){
	this.$el.html(this.template());
	setElement
	this.childView1.setElement("#element1").render();
	this.childView2.setElement("#element2").render();
}
...
```
##Problem :
But sometimes for you parent view you first need you model to be fetched and then render it and also maybe each of your child views need to wait for their model/collection to be fetched and then render their information in DOM.

So then we can't use our ```view.render().el``` and append it to DOM in this kind of situations.

Also you may have some route filtering functions in you router or maybe and showView function which remove the currentView form DOM and loads the new view for you( i'm going to talk about this in future posts) so you need one way for rendering all your views that should work for both views that need to wait for model to be fetched to render and the views that just need to render their simple template in to DOM.

##Solution :

The simple solution that I learned form [Backbone Fundamentals](http://addyosmani.github.io/backbone-fundamentals) and i'm going to expand it a little bit is that we append ``` views.$el``` everywhere in our DOM or in our parent view's ```$el``` . Then anywhere inside out code we just need render the view to place it information into DOM.So inside the view we listen for model/collection to be fetched and when it happened it immediately render the view. Also should noted that using setElement ( ```childView.setElemet('#someElement')```) for child views do something very similar to what we are doing here (```this.$('#someElement').html(this.childView.$el); ```) but you cant assign className and tagName.

``` js

var ParentView = Backbone.View.extend({

	initialize : function () {
		this.listenTo(this.model, 'sync', this.render);
	},

	render : function(){
		this.$el.html(this.template(this.model.toJSON()));
		this.$('#someElement').html(this.childView.$el);
		/*
		* this.$el.append(this.childView.$el);
		*/
		var someModel = new Model;
		this.childView = new ChildView({ model : someModel });
		someModel.fetch();
	}

});

var ChildView = Backbone.View.extend({
	initialize : function(){
		this.listenTo(this.model, 'sync', this.render);
	},

	render : function(){
		this.$el.html(this.template(this.model.toJSON()));
	}
});

```

And for the views that were work fine with ```view.render().el``` now you have to insert ``` this.render()``` in their ```initialize``` method. There are couple of ways for taking care of this views, for example returning the view in showView function and then something like this ``` showView(new SubView).render()``` but putting rendering it in initialize is better I think.

At end something like this can be our showView function in our router:

``` js
. . . 

showView: function (view) {
	/*
	* I'm going talk more about first two lines in future post
	*/
    if (this.currentView) this.currentView.close();
    this.currentView = view;
    $('#page').html(view.$el);
    return view;
},

. . . 

```


That's it.
Hope it would be helpful.


See ya !!!




