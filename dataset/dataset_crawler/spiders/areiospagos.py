from scrapy.spiders import CrawlSpider, Rule, Spider
from scrapy.linkextractors import LinkExtractor
import re
from dataset_crawler.items import AreiosPagosCrawlerItem


def find_duplicates_in_list(l):
    return set([x for x in l if l.count(x) > 1])
def sets_have_duplicates(s1, s2):
    return len(s1.intersection(s2))>0

def find_court_type_from_text(text):
    '''
    finds the court type for an Areios Pagos case-law text
    Input should be a list of strings, each containing a different line from the text
    The court type is expected to be found immediately after the "ΤΟ ΔΙΚΑΣΤΗΡΙΟ ΤΟΥ ΑΡΕΙΟΥ ΠΑΓΟΥ" sentence
    '''
    signifier="ΤΟ ΔΙΚΑΣΤΗΡΙΟ ΤΟΥ ΑΡΕΙΟΥ ΠΑΓΟΥ"
    for idx,sentence in enumerate(text):
        if signifier in sentence.strip():
            if ("ΤΜΗΜΑ" in text[idx+2].strip()) or ("Τμήμα" in text[idx+2].strip()) :
                court_type_sent_idx = idx+2 # in half of the documents this should be +1
            else:
                court_type_sent_idx = idx+1
            break
    try:
        return text[court_type_sent_idx].strip()
    except:
        # court type not found
        return None

class AreiosPagosSpider(Spider):

    name='AreiosPagos'
    allowed_domains=['areiospagos.gr']
    start_urls=['http://www.areiospagos.gr/nomologia/apofaseis.asp']

    d1 = set()


    def parse(self, response):
        '''
        parses http://www.areiospagos.gr/nomologia/apofaseis.asp
        '''
        self.logger.info('~~~~~~~~~~~~~~~~~~~Parse function called on %s', response.url)

        # get topics and case-law per topic links
        topics = response.xpath("//a/text()").getall()
        topic_links = response.xpath("//a/@href").getall()[1:]


        self.d1.update(topic_links)
        for idx,link in enumerate(topic_links):
            try:
                # create metadata info dictionary
                metadata_info = {'topic': topics[idx],}
                # next_page = response.urljoin(link)

                # visit every topic link
                # and parse them with a callback function
                # while providing metadata info kwargs
                # self.logger.info('\x1b[ Parse topic page called on ' + link)

                yield response.follow(link, \
                                   callback=self.parse_topic_page,\
                                   cb_kwargs=dict(metadata=metadata_info))
            except IndexError:
                self.logger.error("Problem indexing " + str(topics) + " list with index " + idx )

    def parse_topic_page(self, response, metadata):
        '''
        - parses case-law topic pages
        (e.g http://www.areiospagos.gr/nomologia/apofaseis_result.asp?s=2&code=585)
        - inserts date and case-law id to metadata dictionary
        '''
        # get link for each case_law
        case_law_links = response.xpath("//table//@href").getall()
        # get (id,year) from each case law
        case_law_info = response.xpath("//table//a//text()").getall()
        # partition list into 2 seperate lists
        case_law_num = [elem.split('/')[0] for elem in case_law_info]
        case_law_year = [elem.split('/')[1] for elem in case_law_info]
        # for each case_law, follow corresponding link and provide cb_kwargs

        if find_duplicates_in_list(case_law_links):
            self.logger.error("duplicate links: " + str(case_law_links))
            raise ValueError



        if sets_have_duplicates(set(case_law_links), self.d1):
            self.logger.error("duplicate links: " + str(set(case_law_links).intersection(self.d1)))
            raise ValueError
            return {}

        for idx,link in enumerate(case_law_links):
            # create unique metadata dictionary for each case_law
            augm_metadata = metadata.copy()
            # insert new metadata
            augm_metadata['year'] = case_law_year[idx]
            augm_metadata['num'] = case_law_num[idx]
            # create next_page url and yield its response's results
            # self.logger.info('\x1b[ Parse case_law page called on ' + link)

            # yield link
            yield response.follow(link, \
                                  callback=self.parse_case_law_page,\
                                  cb_kwargs=dict(metadata=augm_metadata))

    def parse_case_law_page(self, response, metadata):
        '''
        - Parses case-law pages.
        (e.g http://www.areiospagos.gr/nomologia/apofaseis_DISPLAY.asp?cd=XHLHX3G9TZ7XAKQP6MS3GFM7K4S4OF&apof=725_2014&info=%D0%CF%CB%C9%D4%C9%CA%C5%D3%20-%20%20%C3)
        - Creates DatasetCrawlerItem
        '''
        try:
            # get tags corresponding to case-law (e.g Αιτιολογίας επάρκεια, Ακυρότητα απόλυτη, Βούλευμα παραπεμπτικό)
            case_law_tags = response.xpath("//i[b/text() = 'Θέμα']//text()").getall()[1]
            # get case-law summary
            case_law_summary = response.xpath("//p[b/text() = 'Περίληψη:']//text()").getall()[1:]
            # ...join sentences (which are seperated by newlines)
            case_law_summary = " ".join(case_law_summary)
            # ...and remove leading character which is always newline
            case_law_summary = case_law_summary[1:] 
            # get case-law full text
            case_law_text = response.xpath("/html/body/font/p[4]//text()").getall()
        except:
            self.logger.error("problems getting data on link " +response.url)
            case_law_text = response.xpath("/html/body/font/p[3]//text()").getall()
        # remove some newlines
        #------#
        court_category =find_court_type_from_text(case_law_text)
        try:
            # concatenate sentences into string
            case_law_text = "".join(case_law_text)
        except:
            # missing Περίληψη
            case_law_text = response.xpath("/html/body/font/p[3]//text()").getall()
            case_law_text = "".join(case_law_text)
            case_law_summary = None
            self.logger.warning(f"page: {response.url} missing summary")


        #create scrappy Item object
        item = AreiosPagosCrawlerItem()
        item['url'] = response.url
        item['text'] = case_law_text
        item['case_tags'] = case_law_tags
        item['summary'] = case_law_summary
        item['date'] = metadata['year']
        item['court_category'] = court_category
        item['case_category'] = metadata['topic']
        item['case_num'] = metadata['num'] +"/" + metadata['year']

        # self.logger.info('\x1b[ Yielding case_law item ' + str(item))

        yield item
