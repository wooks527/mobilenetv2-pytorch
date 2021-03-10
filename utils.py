import os

def init_file_for_print(id):
    '''Init results.txt file.
    
    Args:
        id (int): id for filename to save print results
    '''
    if not os.path.isdir('results/'):
            os.mkdir('results/')
    fname = f"results/accs_{id}.txt"
    with open(fname, 'w'):
        pass

def set_print_to_file(print, print_to_file, id):
    '''Set print's output stream to the file.
       This codes are referenced by https://stackoverflow.com/a/27622201.
    Args:
        print (func): print function
        print_to_file (bool): whether to save print results into file or not
        id (int): id for filename to save print results
    Returns:
        print (func): original or renewed print function
    '''
    if print_to_file:
        fname = f"results/accs_{id}.txt"
        def file_print(func):
            def wrapped_func(*args, **kwargs):
                kwargs['file'] = open(fname, 'a')
                return func(*args,**kwargs)
            return wrapped_func
        
        return file_print(print)
    return print