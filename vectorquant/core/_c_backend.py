try:
    import vectorquant_c_core
    C_AVAILABLE = True
except ImportError:
    C_AVAILABLE = False
