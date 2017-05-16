'''transforms this closed-world data set to open world:
- adds background data'''

'''
- params: fg, bg
  - later: guess bg based on list
- read both
- transform all fg to fg label
- bg stays

- other method: write to fitting libsvm file
  - -1, +1 as labels
'''
def to_open(foreground, background, target_dir="open_world", binarize=True):
    """read foreground and background directories, combine into target_dir"""
    pass

def _add_background(foreground, name=None, background=None):
    '''@returns a combined instance with background set merged in'''
    if name:
        date = date_of_scenario(name)
        nextbg = min(BGS, key=lambda x: abs(date_of_scenario(x) - f))
        background = counter.all_from_dir(background)
        # search next BG, load to background-var
    foreground['background'] = background['background']
    return foreground

def date_of_scenario(name):
    '''@return date of scenario as datetime.date object
    >>> date_of_scenario('disabled/2016-11-13')
    datetime.date(2016, 11, 13)
    >>> date_of_scenario('disabled/05-12@10')
    datetime.date(2016, 5, 12)
    >>> date_of_scenario('disabled/bridge--2016-07-06')
    datetime.date(2016, 7, 6)
    >>> date_of_scenario('./0.22/10aI--2016-11-04-50-of-100')
    datetime.date(2016, 11, 4)
    >>> date_of_scenario('wtf-pad/bridge--2016-07-05')
    datetime.date(2016, 7, 5)
    '''
    date = name.split('/')[-1]
    if '@' in date:
        date = date.split('@')[0]
    if '--' in date:
        date = date.split('--')[1]
    tmp = [int(x) for x in date.split('-')[:3]]
    if len(tmp) == 2:
        tmp.insert(0, 2016)
    return datetime.date(*tmp)

