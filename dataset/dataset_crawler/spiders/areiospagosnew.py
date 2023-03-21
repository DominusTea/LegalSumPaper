from scrapy.spiders import CrawlSpider, Rule, Spider
from scrapy.linkextractors import LinkExtractor
from scrapy.http import FormRequest
import re
from dataset_crawler.items import AreiosPagosCrawlerItem, AreiosPagosCrawlerItemNew
import json

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



    name='AreiosPagosNew'
    allowed_domains=['areiospagos.gr']
    start_urls=['http://www.areiospagos.gr/nomologia/apofaseis.asp']

    d1 = set()


    def parse(self, response):
        '''
        parses http://www.areiospagos.gr/nomologia/apofaseis.asp?S=1
        with different (year, judgement number) each time
        '''
        # link='http://www.areiospagos.gr/nomologia/apofaseis.asp?S=1'
        link='http://www.areiospagos.gr/nomologia/apofaseis_result.asp?S=1'
        # year_range = range(2016, 2017)# range(1980,2030)
        year_range = range(1980,2023)
        caselaw_id_range = range(1,3001)
        # caselaw_id_range = range(1,3001)
        # self.logger.info('~~~~~~~~~~~~~~~~~~~Parse function called on %s', response.url)

        for year in year_range:
            for case_num in caselaw_id_range:
                metadata_info = {"year": year, "num": case_num, "case_num": str(case_num)+"/"+str(year)}

                headers = {
                    "Content-Type": "application/x-www-form-urlencoded"
                }

                payload = {
                    "X_SUB_TMHMA": "1",
                    "X_TELESTIS_number": "1",
                    "x_number":str(case_num),
                    "X_TMHMA": "5",
                    "X_TELESTIS_ETOS": "1",
                    "x_ETOS": str(year),
                    "submit_krit": "%C1%ED%E1%E6%DE%F4%E7%F3%E7",
                }
                metadata_info['payload'] = json.dumps(payload)

                yield FormRequest(
                    link,
                    #self.start_urls[0],
                    # formxpath='/html/body/font/i/form',
                    # formname='frmsearch',
                    formdata=payload,
                    dont_filter=True,
                    headers=headers,
                    callback=self.parse_year_num_page,
                    cb_kwargs=dict(metadata=metadata_info)
                )


                # yield FormRequest(
                #
                # # yield response.follow(
                #                     link, \
                #                     method="POST",\
                #                     headers=headers,\
                #                    callback=self.parse_year_num_page,\
                #                    cb_kwargs=dict(metadata=metadata_info),
                #                    dont_filter = True,
                #                    body=json.dumps(payload)
                #                    )
            #
            # except IndexError:
            #     self.logger.error("Problem indexing year: " + year )

    def parse_year_num_page(self, response, metadata):
        '''
        parses a search result page
        '''
        self.logger.info("I am in parse+year_num with metadata " + str(metadata))
        self.logger.info("xpath is " +str(response.xpath("/html/body/font/b/text()[3]").getall()))
        # self.logger.info("body is " + str.encode(str(response.body)).decode("utf-8"))
        # with open(f"tmp_html_4_review_{metadata['year']}_{metadata['num']}.html", "w+", encoding='utf-8') as f:
        #     f.write(response.text)

        # self.logger.info("the data is \n"+ str(response.text))
        try:
            # get list of court compositions
            court_comp_lst = response.xpath("/html/body/font/b/text()[3]").getall()
            if len(court_comp_lst) == 0:
                # no results for this query
                yield None
            else:
                # found results. Iterating over (court_comp)
                for court_comp_idx, court_comp in enumerate(court_comp_lst):

                    url_lst = response.xpath(f"/html/body/table[{court_comp_idx+1}]//@href").getall()# response.xpath(f"/html/body/table[{court_comp_idx+1}]/tbody/tr/td[2]/a/@href").getall()
                    # temp save html to review
                    # /html/body/table[1]/tbody/tr/td[2]/a
                    for extracted_url in url_lst:
                        augm_metadata=metadata.copy()
                        augm_metadata["court_composition"] = court_comp
                        augm_metadata['url'] = extracted_url
                        self.logger.info(f"\n\n\n TEST {augm_metadata['url']}")
                        self.logger.info(f"")
                        self.logger.info(f"accessing /html/body/table[{court_comp_idx+1}]/tbody/tr/td[2]/a/@href")
                        # self.logger.info(f"found next link: {augm_metadata['url']}")
                        yield response.follow(augm_metadata['url'],
                                              callback=self.parse_case_law_page,
                                              cb_kwargs=dict(metadata=augm_metadata),
                                              dont_filter=True)
        except Exception as e:

            self.logger.error("problem with getting data at search results page using query: \n - case_num = " + str(metadata["case_num"]) +" year = " +str(metadata["year"]))
            self.logger.error(e)
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
        # pass
        '''
        - Parses case-law pages.
        (e.g http://www.areiospagos.gr/nomologia/apofaseis_DISPLAY.asp?cd=XHLHX3G9TZ7XAKQP6MS3GFM7K4S4OF&apof=725_2014&info=%D0%CF%CB%C9%D4%C9%CA%C5%D3%20-%20%20%C3)
        - Creates DatasetCrawlerItem
        '''
        self.logger.info(f"parsing case_law_page at url {response.url}")
        try:
            case_law_tags = response.xpath("//i[b/text() = 'Θέμα']//text()").getall()
            if len(case_law_tags) ==0:
                case_law_tags=""
            else:
                case_law_tags=case_law_tags[1]
        except:
            self.logger.error("problems getting tags on link " +response.url)

        # try:
        #     # get tags corresponding to case-law (e.g Αιτιολογίας επάρκεια, Ακυρότητα απόλυτη, Βούλευμα παραπεμπτικό)
        #     case_law_tags = response.xpath("//i[b/text() = 'Θέμα']//text()").getall()[1]
        # except:
        #     case_law_tags=""
        #     self.logger.error("problems getting tags on link " +response.url)
        try:
            # get case-law summary
            case_law_summary = response.xpath("//p[b/text() = 'Περίληψη:']//text()").getall()[1:]
        except:
            case_law_summary=""
            self.logger.error("problems getting summary on link " +response.url)
            # yield None
        try:
            # ...join sentences (which are seperated by newlines)
            case_law_summary = " ".join(case_law_summary)
            # ...and remove leading character which is always newline
            case_law_summary = case_law_summary[1:]
        except:

            self.logger.error("problems getting summary joining on link " +response.url)
        try:
            # self.logger.info(f"\n\n\n\n\n\n\n\n\case law summary is {case_law_summary}\n\n\n\n\n")
            # get case-law full text
            if case_law_summary != "" and case_law_tags != "":
                self.logger.info(f"\n\n\n\n\n\n\n\n\INSIDE1\n\n\n\n\n")
                case_law_text = response.xpath("/html/body/font/p[4]//text()").getall()
            else:

                case_law_text_t1 = response.xpath("/html/body/font/p[3]//text()").getall()
                case_law_text_t2 = response.xpath("/html/body/font/p[2]//text()").getall()

                if len(case_law_text_t2) > len(case_law_text_t1):
                    case_law_text=case_law_text_t2
                else:
                    case_law_text=case_law_text_t1
        except:
            self.logger.error("problems getting text on link " +response.url)

        # remove some newlines
        #------#
        court_category =find_court_type_from_text(case_law_text)
        try:
            # concatenate sentences into string
            case_law_text = "".join(case_law_text)
            self.logger.warning(f'condition is {case_law_text.startswith("Περίληψη") or case_law_text.startswith("Περιληψη") or case_law_text.startswith("περίληψη") or case_law_text.startswith("περιληψη")}\n!\n!\n!\n')

            if case_law_text.startswith("Περίληψη") or case_law_text.startswith("Περιληψη") or case_law_text.startswith("περίληψη") or case_law_text.startswith("περιληψη"):
                if '---' in case_law_text and len("\n".join(case_law_text.split('---'))) > 30:
                    case_law_text="\n".join(case_law_text.split("------")[1:]).replace('--','')
                else:
                    case_law_text=""
        except:
            # missing Περίληψη
            self.logger.warning("inside error\n!\n!\n!\n")
            case_law_text = response.xpath("/html/body/font/p[3]//text()").getall()
            case_law_text = "".join(case_law_text)
            if case_law_text.startswith("Περίληψη") or case_law_text.startswith("Περιληψη") or case_law_text.startswith("περίληψη") or case_law_text.startswith("περιληψη"):
                if '---' in case_law_text and len("\n".join(case_law_text.split('---'))) > 30:
                    case_law_text="\n".join(case_law_text.split("------")[1:]).replace('--','')
                else:
                    case_law_text=""
            case_law_summary = None
            self.logger.warning(f"page: {response.url} missing summary")


        #create scrappy Item object
        item = AreiosPagosCrawlerItemNew()
        item['url'] = response.url
        item['text'] = case_law_text.strip()
        item['case_tags'] = case_law_tags.strip()
        item['summary'] = case_law_summary
        item['date'] = metadata['year']
        item['court_category'] = court_category
        item['case_category'] = ""# metadata['topic']
        item['case_num'] = str(metadata['num']) +"/" + str(metadata['year'])
        item['court_composition'] = metadata["court_composition"].strip()

        # self.logger.info('\x1b[ Yielding case_law item ' + str(item))

        yield item
