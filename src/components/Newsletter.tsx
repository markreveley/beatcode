import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

const Newsletter = () => {
  const [email, setEmail] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success("Thanks for subscribing! (ConvertKit integration coming soon)");
    setEmail("");
  };

  return (
    <div className="bg-gradient-to-r from-accent/10 via-primary/10 to-secondary/10 border border-border rounded-lg p-8">
      <h3 className="text-2xl font-bold mb-2">Subscribe to the Newsletter</h3>
      <p className="text-muted-foreground mb-6">
        Get the latest insights on AI agents, music workflows, and creative technology delivered to your inbox.
      </p>
      <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
        <Input
          type="email"
          placeholder="your@email.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="flex-1"
        />
        <Button type="submit" className="bg-gradient-to-r from-accent via-primary to-secondary">
          Subscribe
        </Button>
      </form>
    </div>
  );
};

export default Newsletter;
