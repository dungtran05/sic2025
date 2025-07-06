using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using MyApp.Data;
using MyApp.Models;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace MyApp.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class StudentController : ControllerBase
    {
        private readonly ILogger<StudentController> _logger;
        private readonly MyDbContext _context;

        public StudentController(ILogger<StudentController> logger, MyDbContext context)
        {
            _logger = logger;
            _context = context;
        }
        // GET: api/Student
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Student>>> GetStudents()
        {
            var students = await _context.Students.ToListAsync();
            if (students == null || !students.Any())
            {
                return NotFound("Không tìm thấy sinh viên nào.");
            }
            return Ok(students);
        }
        // POST: api/Student/SaveResult
        [HttpPost("SaveResult")]
        public async Task<IActionResult> SaveResult([FromBody] Student student)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(ModelState);
            }
            _context.Students.Add(student);
            await _context.SaveChangesAsync();

            return Ok("Dữ liệu sinh viên đã được lưu thành công.");
        }
    }
}
