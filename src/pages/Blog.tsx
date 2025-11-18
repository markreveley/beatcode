import { useParams } from "react-router-dom";
import { useEffect } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import MarkdownContent from "@/components/MarkdownContent";
import Newsletter from "@/components/Newsletter";
import { getPostBySlug } from "@/utils/posts";
import { Link } from "react-router-dom";

const Blog = () => {
  const { slug } = useParams<{ slug: string }>();
  const post = slug ? getPostBySlug(slug) : undefined;

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  if (!post) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-12">
          <div className="border-2 border-foreground p-8">
            <div className="text-sm opacity-70 mb-2">ERROR: 404</div>
            <h1 className="text-4xl font-bold mb-4">Post Not Found</h1>
            <p className="mb-6">The blog post you're looking for doesn't exist.</p>
            <Link to="/" className="border-2 border-foreground px-4 py-2 hover:bg-foreground hover:text-background transition-all inline-flex items-center gap-2">
              ← Back to home
            </Link>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <article className="container mx-auto px-4 py-12 max-w-4xl">
          <Link
            to="/"
            className="border-2 border-foreground px-4 py-2 hover:bg-foreground hover:text-background transition-all inline-flex items-center gap-2 mb-8"
          >
            ← Back to all posts
          </Link>

          <header className="mb-8 pb-8 border-b-2 border-foreground">
            <div className="text-xs opacity-70 mb-4">
              <time dateTime={post.frontmatter.date}>
                {new Date(post.frontmatter.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: '2-digit',
                  day: '2-digit'
                }).replace(/\//g, '-')}
              </time>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
              {post.frontmatter.title}
            </h1>
            <div className="border-l-4 border-foreground pl-4 mb-6">
              <div className="text-xs opacity-70 mb-1">SUMMARY:</div>
              <p className="text-lg">
                {post.frontmatter.description}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {post.frontmatter.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs border-2 border-foreground px-3 py-1"
                >
                  {tag}
                </span>
              ))}
            </div>
          </header>

          <div className="mb-12">
            <MarkdownContent content={post.content} />
          </div>

          <div className="border-t-2 border-foreground pt-8">
            <Newsletter />
          </div>
        </article>
      </main>
      <Footer />
    </div>
  );
};

export default Blog;
