import { Link } from "react-router-dom";
import logo from "@/assets/beatcode-logo.png";

const Header = () => {
  return (
    <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
          <img src={logo} alt="beatcode.ai" className="h-10 w-10" />
          <span className="text-2xl font-bold bg-gradient-to-r from-accent via-primary to-secondary bg-clip-text text-transparent">
            beatcode.ai
          </span>
        </Link>
        <nav className="flex items-center gap-6">
          <Link 
            to="/" 
            className="text-foreground hover:text-primary transition-colors font-medium"
          >
            Blog
          </Link>
          <Link 
            to="/about" 
            className="text-foreground hover:text-primary transition-colors font-medium"
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
