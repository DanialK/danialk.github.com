## Setup
- Install rvm
```
curl -L https://get.rvm.io | bash -s stable --ruby
brew install openssl
rvm install 1.9.3 --with-openssl-dir=`brew --prefix openssl`
rvm use 1.9.3
rvm rubygems latest
```
- `gem install bundler -v 1.17.3`
- `bundle install`
- `rake install`
- `virtualenv --python=/usr/bin/python2.7 venv`
- `source venv/bin/activate`
- `rake generate`
- `rake preview`