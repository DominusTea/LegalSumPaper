# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import pymongo
#import settings
from scrapy.utils.project import get_project_settings
import yaml


class MongoDBPipeline(object):
    def __init__(self):
        dbCreds = yaml.safe_load(open('./dbCreds.env'))
        settings = get_project_settings()
        connection = pymongo.MongoClient("mongodb+srv://dbDominusTea:{}@devcluster.qpeoj.mongodb.net/myFirstDatabase?retryWrites=true&w=majority".format(dbCreds['password']))
        db = connection[settings['MONGODB_DB']]
        self.collection = db[settings['MONGODB_COLLECTION']]
    def process_item(self, item, spider):
        valid = True
        for data in item:
            if not data:
                valid = False
                raise DropItem("Missing {0}!".format(data))
        if valid:
            self.collection.insert(dict(item))
#            log.msg("Question added to MongoDB database!",
#                    level=log.DEBUG, spider=spider)
        return item

class DatasetCrawlerPipeline:
    def process_item(self, item, spider):
        return item
