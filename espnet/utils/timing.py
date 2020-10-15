import time

class Timer(object):
    def __init__(self):
        self.times = {}
        return
    def tic(self, key):
        '''
        - Start ticking
        Args:
            key (str): name for ticking
        Returns:
            None
        '''
        if key not in self.times:
            self.times[key] = {'all':0., 'count':0., 'hist': []}
        self.times[key]['t1'] = time.time()
    def toc(self, key):
        '''
        - Stop ticking
        Args:
            key (str): name for ticking
        Returns:
            None
        '''
        self.times[key]['t2'] = time.time()
        t1 = self.times[key]['t1']
        t2 = self.times[key]['t2']
        self.times[key]['hist'].append(t2 -t1)
        self.times[key]['all'] += (t2-t1)
        self.times[key]['count'] += 1
    def __call__(self, key):
        '''
        - Get time
        Args:
            key (str): key used for ticking
        Returns:
            all_time (float): accumulated time (seconds)
        '''
        if not key in self.times:
            return None
        # return self.times[key]['all'] if key in self.times else None
        return ','.join(['%.4f' % x for x in self.times[key]['hist']])
    def avg(self, key):
        '''
        - Get average time
        Args:
            key (str): key used for ticking
        Returns:
            all_time (float): averaged accumulated time (seconds)
        '''
        return self.times[key]['all'] / self.times[key]['count'] if key in self.times else None