try:
    from pkg_resources import resource_filename
except ImportError:
    import os.path

    def resource_filename(module_name, filename):
        """Get the filename for a given resource."""
        mod = __import__(module_name)
        return os.path.join(os.path.dirname(mod.__file__), filename)
