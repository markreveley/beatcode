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
              About stratta.dev
            </h1>

            <div className="border-l-4 border-foreground pl-4 mb-8">
              <p className="text-lg">
                stratta.dev focuses on reliability-first agent architectures built on the BEAM virtual machine, measured and open-sourced.
              </p>
            </div>
          </div>

          {/* Focus Section */}
          <section className="border-2 border-foreground p-6 mb-8">
            <div className="text-sm opacity-70 mb-3">SECTION: OUR FOCUS</div>
            <h2 className="text-2xl font-bold mb-4">Our Focus</h2>
            <p className="leading-relaxed">
              We build and share <strong>production-grade agent systems</strong> on Erlang and Elixir,
              prioritizing fault tolerance, observability, and measurable reliability.
              Every pattern is tested, benchmarked, and open-sourced.
            </p>
          </section>

          {/* What Is the BEAM */}
          <section className="border-2 border-foreground p-6 mb-8">
            <div className="text-sm opacity-70 mb-3">SECTION: TECHNOLOGY</div>
            <h2 className="text-2xl font-bold mb-4">Why the BEAM?</h2>
            <p className="mb-4 leading-relaxed">
              The BEAM virtual machine (Erlang/Elixir) is purpose-built for reliable, concurrent systems.
              For agent architectures, this means:
            </p>
            <div className="space-y-2">
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">01</span>
                <span>Built-in fault tolerance with supervisor trees</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">02</span>
                <span>Lightweight processes for massive agent concurrency</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">03</span>
                <span>Hot code reloading without downtime</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">04</span>
                <span>Built-in observability and telemetry</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-xs border border-foreground px-2 py-1 mt-1">05</span>
                <span>Battle-tested in telecom for 30+ years</span>
              </div>
            </div>
          </section>

          {/* Why This Matters */}
          <section className="border-2 border-foreground p-6 mb-8 hatch-pattern">
            <div className="bg-background p-6 border-2 border-foreground">
              <div className="text-sm opacity-70 mb-3">SECTION: IMPORTANCE</div>
              <h2 className="text-2xl font-bold mb-4">Why Reliability Matters</h2>
              <p className="leading-relaxed">
                Most agent frameworks prioritize rapid prototyping over production readiness.
                We focus on <em>measured reliability</em>—agent systems that handle failures gracefully,
                scale predictably, and provide clear observability into their behavior.
                Every architecture pattern is stress-tested and benchmarked.
              </p>
            </div>
          </section>

          {/* Who This Is For */}
          <section className="border-2 border-foreground p-6 mb-8">
            <div className="text-sm opacity-70 mb-3">SECTION: AUDIENCE</div>
            <h2 className="text-2xl font-bold mb-4">Who This Is For</h2>
            <p className="leading-relaxed">
              Whether you're building production agent systems, exploring fault-tolerant architectures,
              or want to learn how Erlang/Elixir enables reliable AI applications, this resource
              provides tested patterns, performance metrics, and open-source implementations.
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
