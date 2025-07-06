using Microsoft.EntityFrameworkCore;
using MyApp.Models;

namespace MyApp.Data
{
    public class MyDbContext : DbContext
    {
       public MyDbContext(DbContextOptions<MyDbContext> options) : base(options) { }
       

        public DbSet<Student> Students { get; set; } = null!;
        
   
    }
}