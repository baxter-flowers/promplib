from matplotlib import use
use('Agg')

# Agg prevents this exception:
# Exception RuntimeError: RuntimeError('main thread is not in main loop',) in <bound method PhotoImage.__del__ of <Tkinter.PhotoImage instance at 0x7fe8287e87e8>> ignored
# Tcl_AsyncDelete: async handler deleted by the wrong thread
# http://stackoverflow.com/a/29172195/3884647
