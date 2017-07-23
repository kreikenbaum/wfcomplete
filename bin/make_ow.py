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
from scenario import Scenario


def to_open(foreground, background, target_dir="open_world", binarize=True):
    """read foreground and background directories, combine into target_dir"""
    pass

# todo: codup analyse.py
def _add_background(foreground, name=None, background=None):
    '''@returns a combined instance with background set merged in'''
    if name:
        date = Scenario(name).date
        nextbg = min(BGS, key=lambda x: abs(Scenario(x).date - date))
        background = counter.all_from_dir(background)
        # search next BG, load to background-var
    foreground['background'] = background['background']
    return foreground
