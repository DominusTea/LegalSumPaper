# Dataset Crawler
- This folder contains the code used to crawl case-law repositories and store the data found. The craweler was builtusing the [scrapy](https://scrapy.org/) framework.  
Each domain is crawled using a different `spider` object, which can be found in the [spiders folder](spiders).  

- To run each spider:  
```
scrapy crawl SpiderName {-o "path/to/output.[csv|json]"}  
```  
For example, to run the latest version of the AreiosPagos crawler:  
```
scrapy crawl AreiosPagosNew {-o "path/to/output.[csv|json]"}  
*WARNING:*  You should configure the spiders by editing the [settings.py](settings.py) file.  Make sure the configuration is as follows:  
1. ```HTTPCACHE_ENABLED``` is set to ```False```. This makes sure the crawler will download anew each url.  
2. ```AUTOTHROTTLE_ENABLED``` may need to be set to ```True``` in case the server cannot meet the spider's request load.  
3. Crawl responsibly by identifying yourself on the user-agent by defining the ```USER_AGENT``` variable.
4. Enable logging by setting ```LOG_ENABLED``` to ```True``` and defining the output file ```LOG_FILE```. You can, also, set the ```LOG_LEVEL``` variable to one of the following values:  
  1. ```'CRITICAL'```  
  2. ```ERROR'``` (**Recommended**)  
  3. ```'WARNING'```  
  4. ```'INFO'```  
