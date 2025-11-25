import Header from "@/components/Header";
import Footer from "@/components/Footer";
import PostCard from "@/components/PostCard";
import Newsletter from "@/components/Newsletter";
import { getAllPosts } from "@/utils/posts";
const Index = () => {
  const posts = getAllPosts();
  return <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        {/* Hero Section - Technical Diagram Style */}
        <section className="border-b-2 border-foreground">
          <div className="container mx-auto px-4 py-16 md:py-24">
            {/* Title as a technical label */}
            <div className="mb-8">
              <div className="text-sm opacity-70 mb-2">DOMAIN:</div>
              <h1 className="text-4xl md:text-6xl font-bold leading-tight">Audio observability on the BEAM</h1>
            </div>

            {/* Workflow diagram */}
            <div className="max-w-4xl mb-8">
              <div className="text-sm opacity-70 mb-4">WORKFLOW:</div>
              <div className="flex flex-wrap items-center gap-4 text-lg md:text-xl">
                <div className="border-2 border-foreground px-4 py-2">
                  BEAM VM
                </div>
                <span className="text-2xl">→</span>
                <div className="border-2 border-foreground px-4 py-2 hatch-pattern">
                  AGENTS
                </div>
                <span className="text-2xl">→</span>
                <div className="border-2 border-foreground px-4 py-2">
                  RELIABLE SYSTEMS
                </div>
              </div>
            </div>

            {/* Description */}
            <div className="max-w-3xl">
              <div className="text-sm opacity-70 mb-2">DESCRIPTION:</div>
              <p className="text-lg md:text-xl">Evals and testing in Elixir, measured and open-sourced
              <br className="hidden md:block" />
                measured and open-sourced.
              </p>
            </div>
          </div>
        </section>

        {/* Posts Grid */}
        <section className="container mx-auto px-4 py-12">
          <div className="mb-8">
            <div className="text-sm opacity-70 mb-2">LATEST POSTS:</div>
            <div className="flex items-center gap-4">
              <h2 className="text-3xl font-bold">Recent Tutorials</h2>
              <div className="flex-1 h-[2px] bg-foreground"></div>
              <div className="text-sm opacity-70">{posts.length} TOTAL</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {posts.map(post => <PostCard key={post.slug} slug={post.slug} title={post.frontmatter.title} date={post.frontmatter.date} description={post.frontmatter.description} tags={post.frontmatter.tags} />)}
          </div>

          <Newsletter />
        </section>
      </main>
      <Footer />
    </div>;
};
export default Index;
