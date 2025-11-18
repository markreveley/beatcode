import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Newsletter from "@/components/Newsletter";

const About = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <div className="container mx-auto px-4 py-12 max-w-4xl">
          {/* Title Section */}
          <div className="mb-12">
            <div className="text-sm opacity-70 mb-2">DOCUMENTATION:</div>
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              About beatcode.ai
            </h1>

            <div className="border-l-4 border-foreground pl-4 mb-8">
              <p className="text-lg">
                beatcode.ai explores the cutting edge of agentic AI architectures applied to music workflows and creative arts.
              </p>
            </div>
          </div>

          {/* Focus Section */}
          <section className="border-2 border-foreground p-6 mb-8">
            <div className="text-sm opacity-70 mb-3">SECTION: OUR FOCUS</div>
            <h2 className="text-2xl font-bold mb-4">Our Focus</h2>
            <p className="leading-relaxed">
              We dive deep into <strong>Letta agents</strong> and their applications in music production,
              exploring how autonomous AI systems can enhance creativity, streamline workflows,
              and open new possibilities for artists and producers.
            </p>
          </section>

          {/* What Are Letta Agents */}
          <section className="border-2 border-foreground p-6 mb-8">
            <div className="text-sm opacity-70 mb-3">SECTION: TECHNOLOGY</div>
            <h2 className="text-2xl font-bold mb-4">What Are Letta Agents?</h2>
            <p className="mb-4 leading-relaxed">
              Letta is a framework for building stateful, intelligent agents that can maintain context,
              learn from interactions, and execute complex tasks autonomously. In music production,
              these agents can:
            </p>
            <div className="space-y-2">
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">01</span>
                <span>Analyze and categorize audio samples intelligently</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">02</span>
                <span>Suggest musical ideas based on learned patterns</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">03</span>
                <span>Automate repetitive production tasks</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">04</span>
                <span>Assist with mixing and mastering decisions</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">05</span>
                <span>Generate creative variations and alternatives</span>
              </div>
            </div>
          </section>

          {/* Why This Matters */}
          <section className="border-2 border-foreground p-6 mb-8 hatch-pattern">
            <div className="bg-background p-6 border-2 border-foreground">
              <div className="text-sm opacity-70 mb-3">SECTION: IMPORTANCE</div>
              <h2 className="text-2xl font-bold mb-4">Why This Matters</h2>
              <p className="leading-relaxed">
                The intersection of AI and music is rapidly evolving. While AI has made tremendous
                strides in generation and analysis, <em>agentic systems</em> represent the next frontier—
                AI that doesn't just respond to prompts, but actively collaborates, learns, and adapts
                to your creative process.
              </p>
            </div>
          </section>

          {/* Who This Is For */}
          <section className="border-2 border-foreground p-6 mb-8">
            <div className="text-sm opacity-70 mb-3">SECTION: AUDIENCE</div>
            <h2 className="text-2xl font-bold mb-4">Who This Is For</h2>
            <p className="leading-relaxed">
              Whether you're a music producer curious about AI tools, a developer interested in
              creative applications, or an AI researcher exploring agentic architectures, this
              blog aims to provide practical insights, technical deep-dives, and inspiration
              for building the future of music technology.
            </p>
          </section>

          {/* Newsletter */}
          <div className="mt-12">
            <Newsletter />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default About;
