# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DatasetCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    url = scrapy.Field()
    text = scrapy.Field()
    summary = scrapy.Field()
    date = scrapy.Field() # the date of the judicial judgement
    court = scrapy.Field() # the court (e.g Areios Pagos)
    court_category = scrapy.Field() #type of court
    case_category = scrapy.Field() #type of case
    case_num = scrapy.Field()
    pass

class AreiosPagosCrawlerItem(DatasetCrawlerItem):
    # court = scrapy.Field(DatasetCrawlerItem.fields['court'])
    #
    # court = "AreiosPagos"
    case_tags = scrapy.Field()

class AreiosPagosCrawlerItemNew(AreiosPagosCrawlerItem):
    '''
    extended with the court composition field
    '''

    court_composition = scrapy.Field()

class EJusticeCrawlerItem(DatasetCrawlerItem):
    # pass
    metadata_url = scrapy.Field()
    ECLI_provider= scrapy.Field()
