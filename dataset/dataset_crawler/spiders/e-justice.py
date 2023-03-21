from scrapy.spiders import CrawlSpider, Rule, Spider
from scrapy.linkextractors import LinkExtractor
from dataset_crawler.items import EJusticeCrawlerItem

import unidecode

# number of results per year
year_results = {
        2021: 144,
        2020: 2175 ,
        2019: 2994,
        2018: 745,
        2017: 2733,
        2016: 415,
        2015: 808,
        2014: 1486,
        2013: 833,
        2012: 2037,
        2011: 1048,
        2010: 2981,
        2009: 3467,
        2008: 4022,
        2007: 3886,
        2006: 4035,
        2005: 3660,
        2004: 3524,
        2003: 3691,
        2002: 3764,
        2001: 4141,
        2000: 3684,
        1999: 3340,
        1998: 2499,
        1997: 1626,
        1996: 3,
        1995: 3,
        1994: 2,
        1993: 2,
        1989: 19
            }

def remove_accents(str):
    '''
    Removes accents from input greek(!) string.
    Input: str
    Output: str
    '''
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    return normalize('NFD',str).translate(d)

def equal_len_lsts(lsts):
    '''
    Returns true iff the lists have equal length
    Input: [lst1, ..., lstn]
    Output: True/False
    '''
    lengths_lst = map(len, lsts)
    return len(set(lengths_lst)) == 1


class EJusticeSpider(Spider):

    name="E-Justice"
    allowed_domains=['e-justice.europa.eu', 'adjustice.gr']
    start_urls=['https://e-justice.europa.eu/eclisearch/integrated/beta/search.html?text=%22%CE%A3%CE%A5%CE%9C%CE%92%CE%9F%CE%A5%CE%9B%CE%99%CE%9F+%CE%A4%CE%97%CE%A3+%CE%95%CE%A0%CE%99%CE%9A%CE%A1%CE%91%CE%A4%CE%95%CE%99%CE%91%CE%A3%22&ascending=false&sort=DATE&lang=el&text-language=EL&index=25']

    url_set=set()

    def parse(self, response):
        '''
        starts the parsing from  https://e-justice.europa.eu/eclisearch/integrated/beta/search.html?text=%22%CE%A3%CE%A5%CE%9C%CE%92%CE%9F%CE%A5%CE%9B%CE%99%CE%9F+%CE%A4%CE%97%CE%A3+%CE%95%CE%A0%CE%99%CE%9A%CE%A1%CE%91%CE%A4%CE%95%CE%99%CE%91%CE%A3%22&ascending=false&sort=DATE&lang=el&text-language=EL&index=25&type-coded=02
        '''

        N_total_results = response.xpath('//label[@for="Απόφαση"]/em[@class="result-count"]//text()').getall()[0]
        # remove left and right parentheses
        N_total_results=N_total_results[1:len(N_total_results)-1]
        # remove non-breaking spaces (&nbsp) if those exist
        if '\xa0' in N_total_results:
            N_total_results =  N_total_results.replace('\xa0','')
        # case to integer
        N_total_results = int(N_total_results)
        print(N_total_results)
        self.logger.info("Number of Total Results: "+ str(N_total_results))
        for year in year_results:
            N_results = year_results[year]

        # use parse_search_page method on differently indexed page (e.g https://e-justice.europa.eu/eclisearch/integrated/beta/search.html?text=%22%CE%A3%CE%A5%CE%9C%CE%92%CE%9F%CE%A5%CE%9B%CE%99%CE%9F+%CE%A4%CE%97%CE%A3+%CE%95%CE%A0%CE%99%CE%9A%CE%A1%CE%91%CE%A4%CE%95%CE%99%CE%91%CE%A3%22&ascending=false&sort=DATE&lang=el&text-language=EL&type-coded=02&index=50)
            for index in range(1, N_results,25):
                next_link=f"https://e-justice.europa.eu/eclisearch/integrated/beta/search.html?text=%22%CE%A3%CE%A5%CE%9C%CE%92%CE%9F%CE%A5%CE%9B%CE%99%CE%9F+%CE%A4%CE%97%CE%A3+%CE%95%CE%A0%CE%99%CE%9A%CE%A1%CE%91%CE%A4%CE%95%CE%99%CE%91%CE%A3%22&ascending=false&sort=DATE&lang=el&text-language=EL&type-coded=02&year={year}&index={index}"
                self.logger.info("parsing: " + next_link)
                # print("Next_link: ", next_link)
                yield response.follow(next_link,
                                      callback=self.parse_search_page)



    def parse_search_page(self, response):
        '''
        parses https://e-justice.europa.eu/eclisearch/integrated/beta/search.html?text=%22%CE%A3%CE%A5%CE%9C%CE%92%CE%9F%CE%A5%CE%9B%CE%99%CE%9F+%CE%A4%CE%97%CE%A3+%CE%95%CE%A0%CE%99%CE%9A%CE%A1%CE%91%CE%A4%CE%95%CE%99%CE%91%CE%A3%22&ascending=false&sort=DATE&lang=el&text-language=EL&index=25&type-coded=02
        '''
        self.logger.info('~~~~~~~~~~~~~~~~~~~Parse_search_page function called on %s', response.url)
        # print('~~~~~~~~~~~~~~~~~~~Parse function called on %s', response.url)

        case_law_metadata_links = response.xpath('//div[@id="result-container"]//div[@class="expand_block"]/a//@href').getall()
        case_law_links_lst = response.xpath('//div[@id="result-container"]//div[@class="expand_block"]//li[@class="text"]//@href').getall()
        ECLI_providers_lst = response.xpath('//div[@id="result-container"]//div[@class="expand_block"]//li[contains(strong, "Πάροχος ECLI:")]//text()').getall()[1::2]
        countries_lst = response.xpath('//div[@id="result-container"]//div[@class="expand_block"]//li[contains(strong, "Χώρα")]//text()').getall()[1::2]
        courts_lst = response.xpath('//div[@id="result-container"]//div[@class="expand_block"]//li[contains(strong, "Δικαστήριο")]//text()').getall()[1::2]
        dates_lst = response.xpath('//div[@id="result-container"]//div[@class="expand_block"]//li[contains(strong, "Ημερομηνία")]//text()').getall()[1::2]
        ECLIs_lst = response.xpath('//div[@id="result-container"]//ul[@class="expand-list ecli"]//div[@class="label"]//@id').getall()[0::2]

        if not(equal_len_lsts([case_law_metadata_links,
                               case_law_links_lst,
                               ECLI_providers_lst,
                               ECLIs_lst,
                               countries_lst,
                               courts_lst,
                               dates_lst]
                              )):
            raise(ValueError("something wrong with the crawling. Unequal length lists"))


        for element in zip(case_law_metadata_links,
                           case_law_links_lst,
                           ECLI_providers_lst,
                           ECLIs_lst,
                           countries_lst,
                           courts_lst,
                           dates_lst):

            metadata_keys=["metadata_url",
                           "url",
                           "ECLI provider",
                           "ECLI num",
                           "country",
                           "court",
                           "date"
                           ]
            metadata_values=[element[0],
                             element[1],
                             element[2],
                             element[3],
                             element[4],
                             element[5],
                             element[6]]

            metadata = dict(zip(metadata_keys, metadata_values))
            #if "ελλαδα" in remove_accents(metadata["country"]).lower():
            if True and (metadata['url'] not in url_set):
                # add the url to the url_set
                self.url_set.add(metadata['url'])
                # parse the text page if country of origin is Greece
                yield response.follow(metadata["url"],
                                      callback=self.parse_text_page,
                                      cb_kwargs=dict(metadata=metadata))
            else:
                self.logger.error(f"Duplicate visit to url: {metadata['url']}")
                pass

    def parse_text_page(self, response, metadata):
        '''
        - parses case-law text page
        (e.g http://www.adjustice.gr/caselaw/ecli?court=COS&year=2021&ordnumber=0305A309.19E2447)
        - yields EJusticeCrawlerItem Item
        '''
        self.logger.info('~~~~~~~~~~~~~~~~~~~parse_text_page function called on %s', response.url)
        # print('~~~~~~~~~~~~~~~~~~~parse_text_page function called on %s', response.url)

        try:
            case_law_text_selector = response.xpath("//text()")
            case_law_text = case_law_text_selector.getall()[0]
        except IndexError:
            print("ERRRORORO")
            self.logger.error("Could not get any text from url: " + response.url)

        if not len(case_law_text_selector.getall()) == 1:
            raise ValueError("Text was not crawled correctly")


        # create scrappy Item object
        item = EJusticeCrawlerItem()
        item['url'] = response.url
        item['metadata_url'] = metadata['metadata_url']
        item['text'] = case_law_text
        item['date'] = metadata['date']
        item['court'] = metadata['court']
        item['ECLI_provider'] = metadata['ECLI provider']
        item['case_num'] = metadata['ECLI num']

        # print(item)
        yield item
