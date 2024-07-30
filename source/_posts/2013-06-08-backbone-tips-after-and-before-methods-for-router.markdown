---
layout: post
title: "Backbone Tips: After and Before Methods For Router"
date: 2013-06-08 01:04
comments: true
categories: 
- Backbone

---

Hi everybody !!

If you remember my last backbone tip, we added a method called ```showView``` and used it to render all of our views in each route !!

Now we might want to do some operation before and after each route, so its nice to implement these two methods and by using a small hack make them called work !!
``` js

Backbone.Router.prototype.before = function () {};
Backbone.Router.prototype.after = function () {};

Backbone.Router.prototype.route = function (route, name, callback) {
  if (!_.isRegExp(route)) route = this._routeToRegExp(route);
  if (_.isFunction(name)) {
    callback = name;
    name = '';
  }
  if (!callback) callback = this[name];

  var router = this;
  
  Backbone.history.route(route, function(fragment) {
    var args = router._extractParameters(route, fragment);

    router.before.apply(router, arguments);
    callback && callback.apply(router, args);
    router.after.apply(router, arguments);

    router.trigger.apply(router, ['route:' + name].concat(args));
    router.trigger('route', name, args);
    Backbone.history.trigger('route', router, name, args);
  });
  return this;
};

```

So by this two line code , this.before would be execute before actual route callback and this.after after it !

So the router would be like this :
``` js

var Router = Backbone.Router.extend({
  
    routes: {
        '': function(){
        	console.log('INDEX ROUTE');
        }
    },
    before: function () {
        console.log('before');
    },
    after: function () {
        console.log('after');
    },
    

});

var router = new Router();

Backbone.history.start();

```