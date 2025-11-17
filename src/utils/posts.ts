import matter from "gray-matter";

export interface PostFrontmatter {
  title: string;
  date: string;
  description: string;
  tags: string[];
}

export interface Post {
  slug: string;
  frontmatter: PostFrontmatter;
  content: string;
}

// Import all markdown files
const postFiles = import.meta.glob("/src/posts/*.md", { query: "?raw", import: "default", eager: true });

export const getAllPosts = (): Post[] => {
  const posts = Object.entries(postFiles).map(([filepath, content]) => {
    const slug = filepath.replace("/src/posts/", "").replace(".md", "");
    const { data, content: markdown } = matter(content as string);
    
    return {
      slug,
      frontmatter: data as PostFrontmatter,
      content: markdown,
    };
  });

  // Sort by date in reverse chronological order
  return posts.sort((a, b) => 
    new Date(b.frontmatter.date).getTime() - new Date(a.frontmatter.date).getTime()
  );
};

export const getPostBySlug = (slug: string): Post | undefined => {
  const posts = getAllPosts();
  return posts.find((post) => post.slug === slug);
};
