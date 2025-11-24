const Footer = () => {
  return (
    <footer className="border-t-2 border-foreground bg-primary text-primary-foreground mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-6">
          {/* Workflow Diagram */}
          <div className="space-y-2">
            <div className="text-xs opacity-70">WORKFLOW:</div>
            <div className="flex items-center gap-2 text-sm">
              <span className="border px-2 py-1">IDEAS</span>
              <span>→</span>
              <span className="border px-2 py-1">CODE</span>
              <span>→</span>
              <span className="border px-2 py-1 hatch-pattern">OUTPUT</span>
            </div>
          </div>

          {/* Links */}
          <div className="space-y-2">
            <div className="text-xs opacity-70">NAVIGATE:</div>
            <div className="flex gap-4 text-sm">
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline"
              >
                Twitter
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline"
              >
                GitHub
              </a>
              <a
                href="/about"
                className="hover:underline"
              >
                About
              </a>
            </div>
          </div>

          {/* Copyright */}
          <div className="space-y-2">
            <div className="text-xs opacity-70">© {new Date().getFullYear()}</div>
            <div className="text-sm">
              Building reliable agent systems<br />
              on the BEAM platform
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-4 border-t border-primary-foreground/20">
          <div className="text-xs opacity-60">
            Fault-Tolerant Agent Architectures on Erlang/Elixir
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
