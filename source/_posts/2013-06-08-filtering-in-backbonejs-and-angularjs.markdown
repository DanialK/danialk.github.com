---
layout: post
title: "Filtering in BackboneJS and AngularJS"
date: 2013-06-08 20:41
comments: true
categories: 
- Backbone
- Angularjs

---

Hi everyone !!!

I'm working on a backbone project at the moment . I have an collection of data and i wanted to filter through them. Using a bit of backbone code I inject this function to my application. This wasn't a pain at all. However after a bit of playing with AngularJS, I thought it would be cool to show you how filtering works both worlds!!
<!-- more -->
##BackboneJS:

First I add this method to my collection :

``` js

var MyCollection = Backbone.Collection.extend({
    
	model: MyModel,

	search : function(letters){
		if(letters == "") return this;
		var pattern = new RegExp(letters,"gi");
		return _(this.filter(function(data) {
		  	return pattern.test(data.get("name"));
		}));
	}
});

```

using underscore filter method that baked into backbone collections I filter through this collection by using a simple regular expression and testing it with all models name!

Now I have an searchItemView that I can use it's module everywhere in my application that i want to search though collection.

``` js
var searchItemView = Backbone.View.extend({

    template:'<label>Search :</label>'+'<input type="text" id="searchItem" />',

    tagName: "form",

    events : {
        'keyup #searchItem' : 'search'
    },

    initialize : function(options){
        this.collection = options.collection;
        this.parentView = options.parentView;
    },
    
    render: function () {
        this.$el.html(this.template);
        return this;
    },

    search : function(e){
        var letters = $("#searchItem").val();
        var filterd = this.collection.search(letters);
        var parentView = this.parentView;
        parentView.$('#items').empty();
        filterd.each(function(item){
            parentView.addOne(item);
        });
    }
});

```
I just pass the collection and parent view to it and it will filter my collection and re-render each model. It will empty the view and using addOne re-render filtered result. addOne method is a method in my parentView that render each item into collection view.

That's it !!

##AngularJS:

Now i'm going to show you a little magic of angular and it's directives !!
I still love my backbone, but it's so cool :D !!!!

It's all the code :

``` js

<html ng-app>
<head>
    <script src="components/angular/angular.js"></script>   
</head>
<body>

    <div ng-controller='Ctrl'>
        //search model
        <input type="text" ng-model='search'>
        <ul>
            <li ng-repeat='name in names | filter:search'>{{name}}</li>
        </ul>
    </div>
    
    <script type="text/javascript">
        function Ctrl ($scope) {
            $scope.names = ['Danial','John','Jane'];
        }
    </script>

</body>
</html>

```

All these html are there just to show you whats happening there but the magic is when we bind filter directive to search model``` <li ng-repeat='name in names | filter:search'>{{name}}</li> ```.

Thats it !!!
 

