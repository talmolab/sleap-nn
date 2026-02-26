# Config Picker

Interactive tool to generate training configurations from your SLEAP label files.

<div style="position: relative; width: 100%; height: 85vh; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px; overflow: hidden;">
    <iframe
        src="app.html"
        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
        allow="clipboard-write"
        title="SLEAP-NN Config Picker">
    </iframe>
</div>

!!! tip "How to use"
    1. **Upload** your `.slp` or `.pkg.slp` file
    2. **Review** the auto-detected settings and recommendations
    3. **Adjust** parameters as needed
    4. **Download** your configuration file(s)

!!! note "Privacy"
    All processing happens locally in your browser. Your data is never uploaded to any server.

---

## Prefer the CLI?

You can also generate configs using the command line:

```bash
# Interactive TUI
sleap-nn config labels.slp

# Auto-generate with defaults
sleap-nn config labels.slp --auto -o config.yaml
```

See the [Config Generator Guide](../../guides/config-generator.md) for more details.
