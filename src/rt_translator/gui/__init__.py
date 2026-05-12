"""Qt-based GUI for BreezeMate · 微伴.

Submodules:
* ``app``                  -- QApplication entry point (``breezemate-gui``)
* ``main_window``          -- Control panel: device, mode, provider, start/stop
* ``subtitle_window``      -- Frameless floating subtitle overlay
* ``settings_dialog``      -- Provider profile + API key editor
* ``pipeline_controller``  -- QObject that runs the asyncio pipeline in a QThread
* ``signal_sink``          -- Adapts pipeline display events into Qt signals
"""
