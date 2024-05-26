Put geckodriver.exe under /thermofisher
https://github.com/mozilla/geckodriver/releases

To enable Selenium

 ```bash
# Install Dependencies
sudo yum -y update
sudo yum -y install git gtk3 libXinerama.x86_64 cups-libs dbus-glib libXext.x86_64 libXrender.x86_64 libXtst.x86_64 Xvfb

# Install Firefox
cd /usr/local/
sudo wget http://ftp.mozilla.org/pub/firefox/releases/89.0/linux-x86_64/en-US/firefox-89.0.tar.bz2
sudo tar xvjf firefox-89.0.tar.bz2 

# Create symbolic link to firefox
sudo ln -s /usr/local/firefox/firefox /usr/bin/firefox

# Create symbolic link to geckodriver
# Download latest geckodriver binary from https://github.com/mozilla/geckodriver/releases
# Put it in local directory, or install it in a common directory 
# and create a symbolic link from the crawler's working directory:
sudo ln -s ./<Crawler working directory>geckodriver <location of geckodriver>
 ```
