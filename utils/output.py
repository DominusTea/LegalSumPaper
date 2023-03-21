import json

def output_summary(directory, filetype, summary, metadata=[], reference_summary=None):
    f = open(f"{directory}.{filetype}", 'w+', encoding='utf-8')
    if filetype =='txt':
        f.write(summary)
        f.close()
    else:
        try:
            json.dump({'summary': summary, 'metadata':metadata, 'reference_summary':reference_summary},
            f,
            ensure_ascii=False
            )
        except Exception as e:
            print("Error printing summary ", reference_summary, "\nError Message: ", e)
            try:
                json.dump({'summary': summary, 'metadata':[[]], 'reference_summary':reference_summary},
                f,
                ensure_ascii=False
                )
            except Exception as e:
                print("Even empty metadata printing failed with: ", e)
        finally:
            f.close()

def output_summaries(directory, filetype, summary, metadata=[], reference_summary=None):
    '''
    Outputs summary to file according to filetype
    '''
    if isinstance(summary, list):
        # if many summaries are given as input
        # assert output directories are also in a list
        assert(directory, list)

        for dir, sum, meta in zip(directory, summary,metadata):
            output_summary(directory=dir,
                           filetype=filetype,
                           summary=sum,
                           metadata=meta,
                           reference_summary=reference_summary
                           )
    else:
        # only one summary is given as input
        output_summary(directory=directory,
                       filetype=filetype,
                       summary=summary,
                       metadata=metadata,
                       reference_summary=reference_summary
                       )
