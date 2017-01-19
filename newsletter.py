class Newsletter(object):
    def __init__(self, summarizer_list):
        self.summarizer_list = summarizer_list

    def format_article(self, summarizer):
        title = summarizer.title.decode('utf-8').encode('ascii', 'xmlcharrefreplace')
        summary = summarizer.summary.decode('utf-8').encode('ascii', 'xmlcharrefreplace')
        reduction = str(round(summarizer.reduction * 100, 2))
        url = summarizer.url.encode('ascii', 'xmlcharrefreplace')
        summary = summary.replace('\n', '</p><p>')
        summary = '<p>' + summary + '</p>'
        formatted = '''
<h2>{0}</h2>
{1}
<b>Reduction:</b> {2}% of original sentences kept.
<b><a href="{3}">Link to full article</a></b>'''.format(title, summary, reduction, url)
        return formatted

    def construct_email(self):
        pass
