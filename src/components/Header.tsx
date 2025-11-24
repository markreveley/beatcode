import { Link } from "react-router-dom";

const Header = () => {
  return (
    <header className="border-b-2 border-foreground bg-background sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo Section - TOON Style Grid */}
        <Link to="/" className="flex items-center gap-4 hover:opacity-80 transition-opacity">
          <div className="grid grid-cols-2 gap-1">
            <div className="w-4 h-4 bg-foreground"></div>
            <div className="w-4 h-4 border-2 border-foreground"></div>
            <div className="w-4 h-4 border-2 border-foreground"></div>
            <div className="w-4 h-4 bg-foreground"></div>
          </div>
          <span className="text-xl font-bold tracking-tight">
            stratta.dev
          </span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-8">
          <Link
            to="/"
            className="text-foreground hover:bg-foreground hover:text-background px-3 py-1 transition-all border-2 border-transparent hover:border-foreground font-medium"
          >
            Blog
          </Link>
          <Link
            to="/about"
            className="text-foreground hover:bg-foreground hover:text-background px-3 py-1 transition-all border-2 border-transparent hover:border-foreground font-medium"
          >
            About
          </Link>
        </nav>
      </div>
      <div className="h-[2px] bg-foreground"></div>
    </header>
  );
};

export default Header;
