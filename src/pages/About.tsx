import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Newsletter from "@/components/Newsletter";

const About = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <div className="container mx-auto px-4 py-12 max-w-4xl">
          <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-accent via-primary to-secondary bg-clip-text text-transparent">
            About beatcode.ai
          </h1>
          
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <p className="text-xl text-muted-foreground mb-8">
              beatcode.ai explores the cutting edge of agentic AI architectures applied to music workflows and creative arts.
            </p>

            <h2 className="text-2xl font-bold mt-8 mb-4">Our Focus</h2>
            <p>
              We dive deep into <strong>Letta agents</strong> and their applications in music production, 
              exploring how autonomous AI systems can enhance creativity, streamline workflows, 
              and open new possibilities for artists and producers.
            </p>

            <h2 className="text-2xl font-bold mt-8 mb-4">What Are Letta Agents?</h2>
            <p>
              Letta is a framework for building stateful, intelligent agents that can maintain context, 
              learn from interactions, and execute complex tasks autonomously. In music production, 
              these agents can:
            </p>
            <ul>
              <li>Analyze and categorize audio samples intelligently</li>
              <li>Suggest musical ideas based on learned patterns</li>
              <li>Automate repetitive production tasks</li>
              <li>Assist with mixing and mastering decisions</li>
              <li>Generate creative variations and alternatives</li>
            </ul>

            <h2 className="text-2xl font-bold mt-8 mb-4">Why This Matters</h2>
            <p>
              The intersection of AI and music is rapidly evolving. While AI has made tremendous 
              strides in generation and analysis, <em>agentic systems</em> represent the next frontier—
              AI that doesn't just respond to prompts, but actively collaborates, learns, and adapts 
              to your creative process.
            </p>

            <h2 className="text-2xl font-bold mt-8 mb-4">Who This Is For</h2>
            <p>
              Whether you're a music producer curious about AI tools, a developer interested in 
              creative applications, or an AI researcher exploring agentic architectures, this 
              blog aims to provide practical insights, technical deep-dives, and inspiration 
              for building the future of music technology.
            </p>

            <div className="mt-12">
              <Newsletter />
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default About;
